"""Promotion gates and publication failure recovery with isolated model doubles."""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class PromoteReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/32_s10_promote_release.py"
        spec = importlib.util.spec_from_file_location("promote_release_test", path)
        cls.script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.script)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.releases = self.root / "releases"
        self.manifests = self.root / "manifests"
        self.releases.mkdir()
        self.manifests.mkdir()
        self.source = self.root / "trained.joblib"
        self.source.write_bytes(b"trained model fixture, never unpickled")
        self.parent_artifact = self.releases / "s10_production_2026-08-30.joblib"
        self.parent_artifact.write_bytes(b"approved parent fixture")
        self.parent = {
            "release_status": "validated_candidate",
            "artifact_sha256": self.script.sha256_file(self.parent_artifact),
            "metadata": {"primary_model": "ARIMA"},
            "forecast": {"primary_model": "ARIMA"},
        }
        self.parent_path = self.manifests / "2026-08-30.json"
        self.write_parent()
        self.args = argparse.Namespace(
            artifact=self.source, releases_dir=self.releases, manifests_dir=self.manifests,
            dry_run=False, force=False,
        )
        self.forecast = {
            "target_date": "2026-09-13", "point": 6., "p10": 5.9, "p90": 6.1,
            "primary_model": "ARIMA", "fallback_used": False,
            "last_observed_date": "2026-09-06",
        }
        self.health = {"last_date": "2026-09-06", "status": "warning", "warnings": ["beta_floor_pressure"]}
        self.metadata = {"primary_model": "ARIMA", "training_end": "2026-09-06"}
        self.output = self.releases / "s10_production_2026-09-06.joblib"
        self.manifest_output = self.manifests / "2026-09-06.json"

    def write_parent(self):
        self.parent_path.write_text(json.dumps(self.parent), encoding="utf-8")

    def model(self, forecast=None, metadata=None):
        data = copy.deepcopy(self.forecast if forecast is None else forecast)
        health = copy.deepcopy(self.health)
        info = copy.deepcopy(self.metadata if metadata is None else metadata)
        result = SimpleNamespace(**data, as_dict=lambda: copy.deepcopy(data))
        return SimpleNamespace(
            predict_next=lambda: result,
            health=lambda: SimpleNamespace(as_dict=lambda: copy.deepcopy(health)),
            metadata=lambda: copy.deepcopy(info),
        )

    def run_promotion(self, models=None):
        with patch.object(self.script, "_load_model", side_effect=models) as loader, \
                contextlib.redirect_stdout(io.StringIO()):
            if models is None:
                loader.return_value = self.model()
            return self.script.run(self.args)

    def assert_no_publication(self):
        self.assertFalse(self.output.exists())
        self.assertFalse(self.manifest_output.exists())
        self.assertFalse((self.manifests / ".promotion.lock").exists())
        self.assertFalse(list(self.releases.glob(".s10-stage-*")))
        self.assertFalse(list(self.manifests.glob(".s10-stage-*")))
        self.assertEqual(self.parent_artifact.read_bytes(), b"approved parent fixture")

    def test_valid_release_publishes_verified_pair_and_keeps_parent(self):
        self.assertEqual(self.run_promotion(), 0)
        result = json.loads(self.manifest_output.read_text())
        self.assertEqual(self.output.read_bytes(), self.source.read_bytes())
        self.assertEqual(result["artifact_sha256"], self.script.sha256_file(self.output))
        self.assertEqual(result["parent_artifact_sha256"], self.parent["artifact_sha256"])
        self.assertEqual(result["release_status"], "validated_candidate")
        self.assertTrue(all(v for k, v in result["quality_gates"].items() if k != "fallback_used"))
        self.assertFalse(result["quality_gates"]["fallback_used"])
        self.assertEqual(result["health"]["warnings"], ["beta_floor_pressure"])

    def test_nonfinite_nonpositive_and_reversed_intervals_are_blocked(self):
        for column, value in (("point", float("nan")), ("p10", float("inf")),
                              ("p90", float("-inf")), ("p10", 0.), ("p10", 6.2), ("p90", 5.8)):
            with self.subTest(column=column, value=value):
                bad = {**self.forecast, column: value}
                with self.assertRaisesRegex(SystemExit, "forecast_finite|interval_ordered"):
                    self.run_promotion([self.model(bad)])
                self.assert_no_publication()

    def test_fallback_and_primary_switch_are_blocked(self):
        cases = [({**self.forecast, "fallback_used": True}, self.metadata, "fallback_not_used"),
                 ({**self.forecast, "primary_model": "Ridge"}, {**self.metadata, "primary_model": "Ridge"}, "primary_unchanged"),
                 (self.forecast, {**self.metadata, "primary_model": "Ridge"}, "primary_unchanged")]
        for forecast, metadata, gate in cases:
            with self.subTest(gate=gate), self.assertRaisesRegex(SystemExit, gate):
                self.run_promotion([self.model(forecast, metadata)])
            self.assert_no_publication()

    def test_missing_or_unverified_parent_is_blocked(self):
        for change in ({"release_status": "draft"}, {"artifact_sha256": "0" * 64},
                       {"metadata": {}}, {"forecast": {"primary_model": "Ridge"}}):
            original = copy.deepcopy(self.parent)
            self.parent.update(change)
            self.write_parent()
            with self.subTest(change=change), self.assertRaisesRegex(SystemExit, "parent_"):
                self.run_promotion()
            self.assert_no_publication()
            self.parent = original
        self.parent_path.unlink()
        with self.assertRaisesRegex(SystemExit, "release anterior ausente"):
            self.run_promotion()
        self.assert_no_publication()

    def test_duplicate_manifest_keys_are_rejected(self):
        self.parent_path.write_text('{"metadata": {}, "metadata": {}}')
        with self.assertRaisesRegex(SystemExit, "duplicate key"):
            self.run_promotion()
        self.assert_no_publication()

    def test_dates_must_match_and_advance(self):
        for change in ({"target_date": "2026-09-20"}, {"last_observed_date": "2026-09-05"}):
            with self.subTest(change=change), self.assertRaisesRegex(SystemExit, "forecast_dates_consistent"):
                self.run_promotion([self.model({**self.forecast, **change})])
            self.assert_no_publication()
        self.health["last_date"] = "2026-08-23"
        self.metadata["training_end"] = "2026-08-23"
        self.forecast.update(last_observed_date="2026-08-23", target_date="2026-08-30")
        with self.assertRaisesRegex(SystemExit, "release_date_advances"):
            self.run_promotion()
        self.assert_no_publication()

    def test_existing_artifact_is_not_overwritten_even_with_force(self):
        self.output.write_bytes(b"existing release")
        for force in (False, True):
            self.args.force = force
            with self.subTest(force=force), self.assertRaises(SystemExit):
                self.run_promotion()
            self.assertEqual(self.output.read_bytes(), b"existing release")
            self.assertFalse(self.manifest_output.exists())

    def test_existing_manifest_is_not_overwritten(self):
        self.manifest_output.write_text('{"existing": true}')
        with self.assertRaises(SystemExit):
            self.run_promotion()
        self.assertEqual(self.manifest_output.read_text(), '{"existing": true}')
        self.assertFalse(self.output.exists())

    def test_roundtrip_difference_or_loader_failure_leaves_no_release(self):
        changed = {**self.forecast, "point": 6.01}
        for restored in (self.model(changed), RuntimeError("load failed")):
            with self.subTest(restored=type(restored).__name__), self.assertRaises((SystemExit, RuntimeError)):
                self.run_promotion([self.model(), restored])
            self.assert_no_publication()

    def test_copy_corruption_is_detected_against_source_hash(self):
        def corrupt(source, destination):
            Path(destination).write_bytes(b"corrupted copy")

        with patch.object(self.script.shutil, "copy2", side_effect=corrupt), \
                self.assertRaisesRegex(SystemExit, "artifact_integrity_verified"):
            self.run_promotion()
        self.assert_no_publication()

    def test_manifest_publication_failure_rolls_back_only_new_artifact(self):
        link = self.script.os.link

        def fail_manifest(source, destination):
            if destination == self.manifest_output:
                raise OSError("manifest publication failed")
            link(source, destination)

        with patch.object(self.script.os, "link", side_effect=fail_manifest), self.assertRaises(OSError):
            self.run_promotion()
        self.assert_no_publication()

    def test_competing_artifact_is_not_deleted(self):
        def occupied(source, destination):
            Path(destination).write_bytes(b"other publisher")
            raise FileExistsError("already exists")

        with patch.object(self.script.os, "link", side_effect=occupied), self.assertRaises(FileExistsError):
            self.run_promotion()
        self.assertEqual(self.output.read_bytes(), b"other publisher")
        self.assertFalse(self.manifest_output.exists())

    def test_lock_prevents_concurrent_publication(self):
        lock = self.manifests / ".promotion.lock"
        lock.write_text("other process")
        with self.assertRaisesRegex(SystemExit, "outra promocao"):
            self.run_promotion()
        self.assertEqual(lock.read_text(), "other process")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.manifest_output.exists())

    def test_parent_change_during_validation_is_rejected(self):
        calls = []

        def load(*args, **kwargs):
            calls.append(args)
            if len(calls) == 2:
                self.parent["note"] = "changed during validation"
                self.write_parent()
            return self.model()

        with self.assertRaisesRegex(SystemExit, "release anterior mudou"):
            self.run_promotion(load)
        self.assert_no_publication()

    def test_dry_run_validates_roundtrip_without_publishing(self):
        self.args.dry_run = True
        self.assertEqual(self.run_promotion(), 0)
        self.assert_no_publication()
        with self.assertRaisesRegex(SystemExit, "serialization_roundtrip_exact"):
            self.run_promotion([self.model(), self.model({**self.forecast, "point": 6.01})])
        self.assert_no_publication()


if __name__ == "__main__":
    unittest.main()
