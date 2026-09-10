"""Calendar regressions for the horizon experiment, without fetching data."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class HorizonExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/33_s10_horizon_experiment.py"
        spec = importlib.util.spec_from_file_location("horizon_experiment_test", path)
        cls.script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.script)

    def setUp(self):
        self.raw = pd.DataFrame({
            "date": pd.date_range("2020-01-05", periods=90, freq="7D"),
            "price": 5 + np.arange(90) * 0.01,
        })
        self.args = SimpleNamespace(
            monthly_liters=200_000, flexibility_fraction=0.25,
            signal_threshold=0.01, carrying_cost=0.002, random_state=42,
        )

    def predict(self, raw, horizon=1, start=20, minimum=10):
        with patch.object(self.script, "_arima_path", side_effect=lambda history, h: history[-1] + 0.02):
            return self.script.build_horizon_predictions(
                raw, horizon=horizon, start_index=start, min_train=minimum,
            )

    def test_contiguous_input_preserves_previous_alignment_and_policy(self):
        for h in (1, 2, 4, 8, 12):
            with self.subTest(horizon=h):
                result = self.predict(self.raw, horizon=h)
                np.testing.assert_array_equal(result.origin_date, self.raw.date.iloc[20:-h])
                np.testing.assert_array_equal(result.target_date, self.raw.date.iloc[20+h:])
                np.testing.assert_allclose(result.actual, self.raw.price.iloc[20+h:])
                np.testing.assert_allclose(result.persistence, self.raw.price.iloc[20:-h])
                summary = self.script.summarize_predictions(result, h, self.args)
                legacy = self.script.simulate_horizon_prebuy(
                    result.drop(columns="origin_date"), horizon=h,
                    carrying_cost_brl_per_liter_week=self.args.carrying_cost,
                )
                self.assertEqual(summary["procurement_status"], "evaluated")
                self.assertEqual(summary["annualized_savings_brl"], legacy.annualized_savings_brl)
                self.assertEqual(summary["n_scored"], len(result))

    def test_gaps_never_stretch_forecast_horizons(self):
        for h in (1, 2, 4, 8, 12):
            with self.subTest(horizon=h):
                result = self.predict(self.raw.drop(index=[35, 36, 37]), horizon=h)
                self.assertTrue((result.target_date - result.origin_date).eq(pd.Timedelta(weeks=h)).all())
                self.assertTrue(result.target_date.diff().iloc[1:].eq(pd.Timedelta(weeks=1)).all())
                targets = self.raw.drop(index=[35, 36, 37]).set_index("date").price
                np.testing.assert_allclose(result.actual, targets.reindex(result.target_date), equal_nan=True)

    def test_arima_receives_full_calendar_with_missing_history(self):
        histories = []

        def forecast(history, h):
            histories.append(history.copy())
            return history[-1]

        with patch.object(self.script, "_arima_path", side_effect=forecast):
            result = self.script.build_horizon_predictions(
                self.raw.drop(index=15), horizon=4, start_index=30, min_train=10,
            )
        self.assertEqual(len(histories[0]), 31)
        self.assertTrue(all(np.isnan(history[15]) for history in histories))
        for history, origin in zip(histories, result.origin_date):
            position = self.raw.index[self.raw.date.eq(origin)][0]
            expected = self.raw.price.iloc[:position+1].to_numpy().copy()
            expected[15] = np.nan
            np.testing.assert_allclose(history, expected, equal_nan=True)
        # A gap only in training does not disable a complete economic window.
        self.assertEqual(self.script.summarize_predictions(result, 4, self.args)["procurement_status"], "evaluated")

    def test_missing_origin_is_not_forecast_and_missing_target_is_not_imputed(self):
        result = self.predict(self.raw.drop(index=35))
        origin = result.loc[result.origin_date.eq(self.raw.date.iloc[35])].iloc[0]
        target = result.loc[result.target_date.eq(self.raw.date.iloc[35])].iloc[0]
        self.assertTrue(pd.isna(origin.persistence) and pd.isna(origin.arima))
        self.assertTrue(pd.isna(target.actual))
        self.assertTrue(np.isfinite(target.arima))

    def test_future_prices_do_not_change_earlier_predictions(self):
        changed = self.raw.copy()
        changed.loc[50:, "price"] *= 10
        before = self.predict(self.raw, horizon=4)
        after = self.predict(changed, horizon=4)
        mask = before.origin_date < self.raw.date.iloc[50]
        np.testing.assert_array_equal(before.loc[mask, "arima"], after.loc[mask, "arima"])

    def test_missing_weeks_do_not_count_toward_minimum_training(self):
        result = self.predict(self.raw.drop(index=range(3, 9)), start=0, minimum=10)
        self.assertEqual(result.origin_date.iloc[0], self.raw.date.iloc[16])

    def test_removed_and_explicit_nan_weeks_are_equivalent(self):
        explicit = self.raw.copy()
        explicit.loc[35, "price"] = np.nan
        pd.testing.assert_frame_equal(self.predict(explicit), self.predict(self.raw.drop(index=35)))

    def test_calendar_rejects_duplicates_and_off_grid_dates(self):
        duplicate = pd.concat([self.raw, self.raw.iloc[:1]])
        off_grid = self.raw.copy()
        off_grid.loc[35, "date"] += pd.Timedelta(days=1)
        for raw in (duplicate, off_grid):
            with self.subTest(kind=len(raw)), self.assertRaises(ValueError):
                self.predict(raw)

    def test_rejects_invalid_parameters_and_observed_prices(self):
        for kwargs in ({"horizon": 0}, {"horizon": 1.5}, {"start": -1}, {"minimum": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.predict(self.raw, **kwargs)
        for value in (-1, 0, np.inf, "invalid"):
            raw = self.raw.astype({"price": object})
            raw.loc[35, "price"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.predict(raw)

    def test_gapped_evaluation_reports_unavailable_economics_without_replay(self):
        predictions = self.predict(self.raw.drop(index=35))
        with patch.object(self.script, "simulate_horizon_prebuy", side_effect=AssertionError("must not compress gaps")):
            result = self.script.summarize_predictions(predictions, 1, self.args)
        self.assertEqual(result["procurement_status"], "unavailable_missing_observations")
        self.assertEqual(result["n_unscored"], 2)
        self.assertEqual(result["n_scored"], len(predictions)-2)
        self.assertTrue(np.isfinite(result["mae"]))
        self.assertIsNone(result["annualized_savings_brl"])
        self.assertIsNone(result["ci90_positive"])

    def test_insufficient_data_and_zero_baseline_error_are_explicit(self):
        empty = self.predict(self.raw, start=100)
        result = self.script.summarize_predictions(empty, 1, self.args)
        self.assertEqual(result["procurement_status"], "unavailable_insufficient_predictions")
        self.assertIsNone(result["mae"])
        constant = self.raw.assign(price=5.)
        result = self.script.summarize_predictions(self.predict(constant), 1, self.args)
        self.assertEqual(result["persistence_mae"], 0.)
        self.assertIsNone(result["mae_ratio_vs_persistence"])
        json.dumps(result, allow_nan=False)

    def test_run_preserves_holdout_boundary_and_serializes_missing_evidence(self):
        raw = self.raw.copy()
        raw["date"] = pd.date_range(end="2024-10-20", periods=len(raw), freq="7D")
        raw = raw.drop(index=35)
        seen = []
        build = self.script.build_horizon_predictions

        def capture(panel, **kwargs):
            self.assertLess(panel.date.max(), pd.Timestamp(self.script.S10_HOLDOUT_START))
            result = build(panel, **kwargs)
            self.assertTrue(result.target_date.lt(pd.Timestamp(self.script.S10_HOLDOUT_START)).all())
            seen.append(result)
            return result

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(**vars(self.args), data=Path(tmp)/"unused", window="development",
                                   horizons=[1, 4], start_index=20, min_train=10, output_dir=Path(tmp))
            with patch.object(self.script, "load_anp_fuel_csv", return_value=raw), \
                    patch.object(self.script, "_arima_path", side_effect=lambda hist, h: hist[-1]), \
                    patch.object(self.script, "build_horizon_predictions", side_effect=capture), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = self.script.run(args)
            self.assertEqual(len(seen), 2)
            self.assertFalse(result["holdout_reopened"])
            self.assertEqual(result["n_missing_weeks"], 1)
            self.assertIsNone(result["best_horizon"])
            json.dumps(result, allow_nan=False)
            saved = json.loads((Path(tmp)/"horizon_experiment.json").read_text())
            self.assertEqual(saved["pipeline_version"], "horizon-calendar-v2")

    @unittest.skipUnless(importlib.util.find_spec("statsmodels"), "requires optional statsmodels")
    def test_real_arima_accepts_missing_weeks(self):
        history = self.raw.price.to_numpy().copy()
        history[35:38] = np.nan
        point = self.script._arima_path(history, 4)
        self.assertTrue(np.isfinite(point) and point > 0)


if __name__ == "__main__":
    unittest.main()
