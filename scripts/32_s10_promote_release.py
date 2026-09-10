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
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.anp_official import sha256_file  # noqa: E402
from vs_epl_krls.production import S10ProductionForecaster  # noqa: E402


def _latest_release_manifest(directory: Path) -> Path | None:
    manifests = sorted(p for p in directory.glob("*.json") if p.stem[:4].isdigit())
    return manifests[-1] if manifests else None


def run(args: argparse.Namespace) -> int:
    artifact = args.artifact
    if not artifact.is_file():
        raise SystemExit(
            f"artefato ausente: {artifact}\n"
            "rode antes: python scripts/06_train_s10_production.py"
        )

    source_hash = sha256_file(artifact)
    model = S10ProductionForecaster.load(artifact, expected_sha256=source_hash)
    forecast = model.predict_next()
    health = model.health().as_dict()
    label = str(health["last_date"])  # semana de competencia da ultima observacao

    releases_dir = args.releases_dir
    manifests_dir = args.manifests_dir
    prior = _latest_release_manifest(manifests_dir)
    parent_hash = None
    if prior is not None:
        parent_hash = json.loads(prior.read_text(encoding="utf-8")).get("artifact_sha256")

    out_artifact = releases_dir / f"s10_production_{label}.joblib"
    out_manifest = manifests_dir / f"{label}.json"

    print(f"artefato treinado : {artifact.name}  ({source_hash[:16]}...)")
    print(f"ultima observacao : {label}")
    print(f"previsao pendente : {forecast.target_date}  R$ {forecast.point:.4f}/L")
    print(f"release destino   : {out_artifact.name}")
    if prior is not None:
        print(f"release anterior  : {prior.name}  (parent {str(parent_hash)[:16]}...)")

    if out_artifact.exists() and not args.force:
        raise SystemExit(
            f"release ja existe: {out_artifact}\n"
            "releases sao imutaveis; use --force apenas se souber o que esta fazendo"
        )

    if args.dry_run:
        print("\n--dry-run: nada foi escrito")
        return 0

    releases_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, out_artifact)
    artifact_hash = sha256_file(out_artifact)

    # Uma release so vale se reler os bytes gravados devolver a mesma previsao.
    restored = S10ProductionForecaster.load(out_artifact, expected_sha256=artifact_hash)
    following = restored.predict_next()
    if following.as_dict() != forecast.as_dict():
        out_artifact.unlink(missing_ok=True)
        raise SystemExit("roundtrip de serializacao mudou a previsao; release descartada")

    payload: dict[str, object] = {
        "release_contract_version": "1.0",
        "release_status": "validated_candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(out_artifact.resolve()),
        "artifact_sha256": artifact_hash,
        "parent_artifact_sha256": parent_hash,
        "forecast": following.as_dict(),
        "health": restored.health().as_dict(),
        "metadata": restored.metadata(),
        "quality_gates": {
            "artifact_integrity_verified": True,
            "serialization_roundtrip_exact": True,
            "forecast_finite": all(
                value == value and abs(value) != float("inf")
                for value in (following.point, following.p10, following.p90)
            ),
            "interval_ordered": 0 < following.p10 <= following.point <= following.p90,
            "fallback_used": following.fallback_used,
        },
        "promotion_note": (
            f"Origem dos dados avancada para a semana de {label}; o modelo primario "
            "permanece o aprovado pela selecao. O holdout nao foi reaberto "
            "(S10_HOLDOUT_END continua congelado) e nenhum challenger foi promovido."
        ),
    }
    out_manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print("\nrelease publicada")
    print(f"  sha256    {artifact_hash}")
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
    parser.add_argument("--force", action="store_true", help="sobrescreve uma release existente")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
