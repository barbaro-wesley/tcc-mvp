"""Promove o bundle de producao ja treinado a uma release imutavel servida pela API.

O que este script faz e, sobretudo, o que ele NAO faz
-----------------------------------------------------
Ele avanca apenas a *origem dos dados* da release: copia o artefato treinado por
``06_train_s10_production.py`` para ``artifacts/releases/``, verifica o SHA-256,
confirma que reler os bytes devolve exatamente a mesma previsao e escreve o
manifesto encadeado ao pai (``parent_artifact_sha256``).

Ele nao reabre o holdout (``S10_HOLDOUT_END`` continua congelado em
selection.py), nao roda selecao e nao troca o modelo primario.  O primario
permanece o que a selecao ja havia aprovado -- tipicamente ARIMA.  Promover um
challenger e outra decisao, com gates e revisao humana, e nao acontece aqui.

Uso::

    python scripts/32_s10_promote_release.py
    python scripts/32_s10_promote_release.py --dry-run
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.anp_official import sha256_file  # noqa: E402


def _load_model(path: Path, expected_sha256: str):
    from vs_epl_krls.production import S10ProductionForecaster

    return S10ProductionForecaster.load(path, expected_sha256=expected_sha256)


def _read_manifest(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def invalid_constant(token):
        raise ValueError(f"non-finite value: {token}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                           parse_constant=invalid_constant)
        if not isinstance(value, dict):
            raise ValueError("manifest must be an object")
        return value
    except (OSError, ValueError) as exc:
        raise SystemExit(f"manifesto invalido: {path}: {exc}") from exc


def _require_gates(gates: dict[str, bool]) -> None:
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise SystemExit("promocao bloqueada: " + ", ".join(failed))


def _forecast_gates(forecast, health: dict, metadata: dict, primary: str) -> dict[str, bool]:
    values = (forecast.point, forecast.p10, forecast.p90)
    finite = all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
    try:
        last_date = date.fromisoformat(health["last_date"])
        dates_match = (
            last_date.isoformat() == health["last_date"]
            == forecast.last_observed_date == metadata.get("training_end")
            and forecast.target_date == (last_date + timedelta(weeks=1)).isoformat()
        )
    except (KeyError, TypeError, ValueError):
        dates_match = False
    return {
        "forecast_finite": finite,
        "interval_ordered": bool(finite and 0 < forecast.p10 <= forecast.point <= forecast.p90),
        "fallback_not_used": forecast.fallback_used is False,
        "primary_unchanged": metadata.get("primary_model") == forecast.primary_model == primary,
        "forecast_dates_consistent": dates_match,
    }


def _latest_release_manifest(directory: Path) -> Path | None:
    manifests = sorted(p for p in directory.glob("*.json") if p.stem[:4].isdigit())
    return manifests[-1] if manifests else None


def run(args: argparse.Namespace) -> int:
    if args.force:
        raise SystemExit("--force nao e permitido: releases publicadas sao imutaveis")
    artifact = args.artifact
    if not artifact.is_file():
        raise SystemExit(
            f"artefato ausente: {artifact}\n"
            "rode antes: python scripts/06_train_s10_production.py"
        )

    source_hash = sha256_file(artifact)
    model = _load_model(artifact, expected_sha256=source_hash)
    forecast = model.predict_next()
    health = model.health().as_dict()
    metadata = model.metadata()

    releases_dir = args.releases_dir
    manifests_dir = args.manifests_dir
    prior = _latest_release_manifest(manifests_dir)
    if prior is None:
        raise SystemExit("release anterior ausente: este comando apenas avanca uma release aprovada")
    prior_manifest_hash = sha256_file(prior)
    parent = _read_manifest(prior)
    parent_metadata = parent.get("metadata")
    primary = parent_metadata.get("primary_model") if isinstance(parent_metadata, dict) else None
    parent_hash = parent.get("artifact_sha256")
    parent_artifact = releases_dir / f"s10_production_{prior.stem}.joblib"
    parent_forecast = parent.get("forecast", {})
    if not isinstance(parent_forecast, dict):
        parent_forecast = {}
    _require_gates({
        "parent_validated": parent.get("release_status") == "validated_candidate",
        "parent_primary_known": primary in {"ARIMA", "Ridge", "persistencia", "VS-ePL-KRLS", "ensemble"},
        "parent_primary_consistent": parent_forecast.get("primary_model") == primary,
        "parent_integrity_verified": isinstance(parent_hash, str) and parent_artifact.is_file()
        and sha256_file(parent_artifact) == parent_hash,
    })
    gates = _forecast_gates(forecast, health, metadata, primary)
    _require_gates(gates)
    label = health["last_date"]
    try:
        elapsed_days = (date.fromisoformat(label) - date.fromisoformat(prior.stem)).days
    except ValueError as exc:
        raise SystemExit("data da release anterior invalida") from exc
    _require_gates({"release_date_advances": elapsed_days > 0 and elapsed_days % 7 == 0})

    out_artifact = releases_dir / f"s10_production_{label}.joblib"
    out_manifest = manifests_dir / f"{label}.json"

    print(f"artefato treinado : {artifact.name}  ({source_hash[:16]}...)")
    print(f"ultima observacao : {label}")
    print(f"previsao pendente : {forecast.target_date}  R$ {forecast.point:.4f}/L")
    print(f"release destino   : {out_artifact.name}")
    if prior is not None:
        print(f"release anterior  : {prior.name}  (parent {str(parent_hash)[:16]}...)")

    if out_artifact.exists() or out_manifest.exists():
        raise SystemExit(
            f"release ja existe: {out_artifact} ou {out_manifest}; releases sao imutaveis"
        )

    payload: dict[str, object] = {
        "release_contract_version": "1.0",
        "release_status": "validated_candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(out_artifact.resolve()),
        "artifact_sha256": source_hash,
        "parent_artifact_sha256": parent_hash,
        "forecast": forecast.as_dict(),
        "health": health,
        "metadata": metadata,
        "promotion_note": (
            f"Origem dos dados avancada para a semana de {label}; o modelo primario "
            "permanece o aprovado pela selecao. O holdout nao foi reaberto "
            "(S10_HOLDOUT_END continua congelado) e nenhum challenger foi promovido."
        ),
    }
    # Os temporarios ficam no mesmo filesystem de cada destino para publicar
    # com hard links atomicos e exclusivos. O manifesto e o ultimo a aparecer.
    # No dry-run a copia e validada no temporario do sistema, sem gravar destinos.
    stage_root = None if args.dry_run else releases_dir
    with tempfile.TemporaryDirectory(prefix=".s10-stage-", dir=stage_root) as staging:
        staged_artifact = Path(staging) / "model.joblib"
        shutil.copy2(artifact, staged_artifact)
        gates["artifact_integrity_verified"] = sha256_file(staged_artifact) == source_hash
        _require_gates(gates)
        restored = _load_model(staged_artifact, expected_sha256=source_hash)
        following = restored.predict_next()
        restored_health = restored.health().as_dict()
        restored_metadata = restored.metadata()
        gates.update(_forecast_gates(following, restored_health, restored_metadata, primary))
        gates["serialization_roundtrip_exact"] = following.as_dict() == forecast.as_dict()
        _require_gates(gates)
        payload.update(forecast=following.as_dict(), health=restored_health, metadata=restored_metadata)
        payload["quality_gates"] = {**gates, "fallback_used": following.fallback_used}
        text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        if args.dry_run:
            print("\n--dry-run: gates e copia validados; nenhuma release publicada")
            return 0

        # Serializa publicacoes deste comando, inclusive de datas diferentes.
        # Um lock deixado por interrupcao deve ser inspecionado pelo operador.
        lock = manifests_dir / ".promotion.lock"
        try:
            handle = lock.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise SystemExit("outra promocao esta em andamento: .promotion.lock") from exc
        try:
            with handle:
                handle.write(str(os.getpid()))
            if _latest_release_manifest(manifests_dir) != prior or sha256_file(prior) != prior_manifest_hash:
                raise SystemExit("release anterior mudou durante a validacao; execute novamente")
            with tempfile.TemporaryDirectory(prefix=".s10-stage-", dir=manifests_dir) as manifest_stage:
                staged_manifest = Path(manifest_stage) / "manifest.json"
                staged_manifest.write_text(text, encoding="utf-8")
                artifact_published = False
                try:
                    os.link(staged_artifact, out_artifact)
                    artifact_published = True
                    os.link(staged_manifest, out_manifest)
                except Exception:
                    if artifact_published:
                        out_artifact.unlink()
                    raise
        finally:
            lock.unlink()

    print("\nrelease publicada")
    print(f"  sha256    {source_hash}")
    print(f"  manifesto {out_manifest}")
    print(f"  gates     {payload['quality_gates']}")
    print("\nsirva com:")
    print(
        f"  python scripts/15_s10_service.py --state RS \\\n"
        f"    --artifact {out_artifact} \\\n"
        f"    --manifest {out_manifest}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, default=ROOT / "artifacts" / "s10_production.joblib"
    )
    parser.add_argument(
        "--releases-dir", type=Path, default=ROOT / "artifacts" / "releases"
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_product" / "releases",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="obsoleto: sobrescrita de releases e bloqueada")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
