"""Treina o modelo de paridade em todo o historico e emite a previsao operacional.

Grava um artefato imutavel verificado por SHA-256, no mesmo espirito do bundle
existente, e imprime a previsao da proxima semana junto com a recomendacao da
politica de antecipacao de compra.

Uso::

    python scripts/21_s10_ingest_causal.py        # atualiza as fontes
    python scripts/23_s10_parity_production.py    # treina e preve
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.audit import (  # noqa: E402
    record_forecast,
    settle_pending_forecast as settle_forecast,
    verify_audit_ledger,
)
from vs_epl_krls.calibration import backtest_adaptive_interval  # noqa: E402
from vs_epl_krls.gates import interval_report  # noqa: E402
from vs_epl_krls.passthrough import (  # noqa: E402
    PARITY_FEATURES,
    PassThroughConfig,
    PassThroughECM,
    build_parity_panel,
    build_next_parity_row,
)

CONTRACT_VERSION = "parity-1.0.0"
SIGNAL_THRESHOLD = 0.01
DEFAULT_MONTHLY_LITERS = 200_000.0
FLEXIBILITY = 0.25
#: Semanas prospectivas exigidas antes de qualquer decisao de promocao.  O
#: holdout ja foi lido duas vezes; daqui em diante so semanas futuras decidem.
PROSPECTIVE_TARGET = 26

#: Semanas de walk-forward causal usadas para calibrar o nivel do intervalo.
#: O modelo estima a *escala* condicional bem, mas o nivel nominal fixo entrega
#: cobertura de 89,4% para um intervalo de 80% no holdout publicado: banda larga
#: demais, que infla o cenario P90 e distorce a decisao de compra.
CALIBRATION_WEEKS = 156


def repo_relative(path: Path) -> str:
    """Caminho portatil para dentro da evidencia versionada.

    O manifesto anterior gravava o caminho absoluto da maquina que treinou o
    modelo, o que vazava o usuario local e impedia conferir a proveniencia em
    outro clone.
    """

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(panel: pd.DataFrame) -> str:
    payload = panel[["date", "price"]].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def calibrated_band(
    model: PassThroughECM, panel: pd.DataFrame
) -> tuple[object, dict[str, object]]:
    """Aprende o nivel do intervalo online sobre o passado recente.

    A escala condicional continua sendo a do modelo; o que a inferencia conformal
    adaptativa ajusta e o *nivel* de miscobertura, que o quantil fixo erra.  Tudo
    e causal: cada banda e emitida antes de a semana correspondente ser vista.
    """

    start = max(1, len(panel) - CALIBRATION_WEEKS)
    history = model.walk_forward(panel, start, len(panel))
    usable = history.dropna(subset=["actual", "prediction", "sigma"])
    if len(usable) < 20:
        return None, {"available": False, "reason": "historico insuficiente para calibrar"}
    result = backtest_adaptive_interval(
        usable["actual"].to_numpy(float),
        usable["prediction"].to_numpy(float),
        scale=usable["sigma"].to_numpy(float),
        nominal_coverage=model.config.interval_nominal,
        gamma=0.02,
        window=CALIBRATION_WEEKS,
        min_residuals=20,
    )
    published = interval_report(
        usable["actual"].to_numpy(float),
        usable["lower"].to_numpy(float),
        usable["upper"].to_numpy(float),
        nominal_coverage=model.config.interval_nominal,
    )
    calibrated = interval_report(
        usable["actual"].to_numpy(float),
        result["lower"],
        result["upper"],
        nominal_coverage=model.config.interval_nominal,
    )
    diagnostics = {
        "available": True,
        "calibration_weeks": int(len(usable)),
        "fixed_quantile": published.as_dict(),
        "adaptive_conformal": calibrated.as_dict(),
        "winkler_improvement": round(published.mean_winkler - calibrated.mean_winkler, 6),
        "width_reduction_fraction": round(
            1.0 - calibrated.mean_width / published.mean_width, 6
        ),
        "state": result["state"].as_dict(),
    }
    return result["band"], diagnostics


def settle_pending_forecast(ledger: Path, panel: pd.DataFrame) -> dict | None:
    """Adapta o painel para o liquidador compartilhado do modulo de auditoria."""

    observed = {
        str(pd.Timestamp(date).date()): float(price)
        for date, price in zip(panel["date"], panel["price"])
        if pd.notna(price)
    }
    return settle_forecast(ledger, observed, model_name="paridade")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", type=Path, default=ROOT / "data" / "processed" / "s10_causal_panel.csv"
    )
    parser.add_argument(
        "--artifact", type=Path, default=ROOT / "artifacts" / "s10_parity.joblib"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports" / "vs_epl_krls" / "s10_parity"
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_parity" / "parity_ledger.jsonl",
    )
    parser.add_argument("--monthly-liters", type=float, default=DEFAULT_MONTHLY_LITERS)
    args = parser.parse_args()
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    causal = pd.read_csv(args.panel, parse_dates=["date"])
    panel = build_parity_panel(causal)
    config = PassThroughConfig()

    settled = settle_pending_forecast(args.ledger, panel)

    model = PassThroughECM(config=config, feature_names=PARITY_FEATURES).fit(panel)
    summary = model.summary()

    # A linha futura usa a mesma construcao causal dos atributos do treino.
    last = panel.iloc[-1]
    origin_price = float(last["price"])
    origin_date = pd.Timestamp(last["date"])
    target_date = origin_date + pd.Timedelta(days=7)

    next_row = build_next_parity_row(causal, config=config)
    forecast = model.forecast_row(next_row, origin_price=origin_price)

    band, calibration = calibrated_band(model, panel)
    forecast_payload = forecast.as_dict()
    if band is not None:
        low, high = band.interval(forecast.point, scale=forecast.conditional_sigma)
        # O nivel nominal fixo fica registrado ao lado, para que a troca seja
        # auditavel em vez de silenciosa.
        forecast_payload["fixed_quantile_lower"] = round(float(forecast.lower), 6)
        forecast_payload["fixed_quantile_upper"] = round(float(forecast.upper), 6)
        forecast_payload["lower"] = round(float(low), 6)
        forecast_payload["upper"] = round(float(high), 6)
        forecast_payload["interval_method"] = "adaptive_conformal_normalized"
    else:
        forecast_payload["interval_method"] = "fixed_quantile"

    expected_rise = forecast.point - origin_price
    recommend = bool(expected_rise > SIGNAL_THRESHOLD)
    weekly_liters = args.monthly_liters * 12.0 / 52.0
    liters = weekly_liters * FLEXIBILITY
    payload = {
        "contract_version": CONTRACT_VERSION,
        "generated_from_panel": repo_relative(args.panel),
        "data_fingerprint": fingerprint(panel),
        "n_train": summary["n_train"],
        "features": list(PARITY_FEATURES),
        "coefficients": summary["coefficients"],
        "forecast": forecast_payload,
        "interval_calibration": calibration,
        "decision": {
            "policy": "antecipar 25% de uma semana quando a alta prevista supera R$ 0,01/L",
            "signal_threshold_brl_per_liter": SIGNAL_THRESHOLD,
            "expected_change_brl_per_liter": round(expected_rise, 6),
            "recommend_prebuy": recommend,
            "liters_if_triggered": round(liters, 1) if recommend else 0.0,
            "expected_saving_brl": round(expected_rise * liters, 2) if recommend else 0.0,
        },
    }

    joblib.dump(
        {"model": model, "config": asdict(config), "payload": payload},
        args.artifact,
        compress=3,
    )
    payload["artifact_sha256"] = sha256_file(args.artifact)
    payload["artifact_bytes"] = args.artifact.stat().st_size

    (args.output_dir / "latest_forecast.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    forecast_record = record_forecast(
        args.ledger, payload, target_date=str(target_date.date())
    )
    ledger_records = verify_audit_ledger(args.ledger)
    realized = [record for record in ledger_records if record["event"] == "realized"]

    print(f"treino: {summary['n_train']} semanas ate {origin_date.date()}")
    print("coeficientes:")
    for name, value in summary["coefficients"].items():
        print(f"  {name:12s} {value:+.6f}")
    print(f"\norigem  {origin_date.date()}  R$ {origin_price:.4f}/L")
    print(f"alvo    {target_date.date()}")
    print(f"ponto   R$ {forecast.point:.4f}/L  (variacao {expected_rise:+.4f})")
    print(f"P10-P90 R$ {forecast_payload['lower']:.4f} a {forecast_payload['upper']:.4f}/L "
          f"({forecast_payload['interval_method']})")
    if calibration.get("available"):
        fixed = calibration["fixed_quantile"]
        adaptive = calibration["adaptive_conformal"]
        print(f"  quantil fixo   cobertura {fixed['empirical_coverage']:.3f} "
              f"largura {fixed['mean_width']:.4f} winkler {fixed['mean_winkler']:.4f}")
        print(f"  conformal      cobertura {adaptive['empirical_coverage']:.3f} "
              f"largura {adaptive['mean_width']:.4f} winkler {adaptive['mean_winkler']:.4f}")
        print(f"  banda {calibration['width_reduction_fraction']:+.1%} mais estreita "
              f"em {calibration['calibration_weeks']} semanas de calibracao")
    print(f"fallback: {forecast.fallback_used}  {forecast.reason}")
    print(f"\ndecisao: {'ANTECIPAR' if recommend else 'NAO ANTECIPAR'} "
          f"({args.monthly_liters:,.0f} L/mes)")
    if recommend:
        print(f"  comprar {liters:,.0f} L uma semana antes; "
              f"economia esperada R$ {expected_rise * liters:,.2f}")
    print(f"\nartefato {args.artifact}")
    print(f"sha256   {payload['artifact_sha256']}")

    if settled is not None:
        scored = settled["payload"]
        print()
        print(f"semana liquidada {scored['target_date']}: "
              f"observado R$ {scored['observed_price']:.4f}/L, "
              f"erro paridade R$ {scored['absolute_error']:.4f}/L, "
              f"erro persistencia R$ {scored['persistence_absolute_error']:.4f}/L, "
              f"intervalo {'cobriu' if scored['interval_covered'] else 'NAO cobriu'}")
    if forecast_record is None:
        print()
        print("ledger: previsao identica ja registrada para esta semana-alvo")
    elif forecast_record["event"] == "forecast_revision":
        print()
        print("ledger: revisao registrada; o artefato desta semana-alvo mudou")
    print(f"ledger   {args.ledger}")
    print(f"registros {len(ledger_records)}, "
          f"head {ledger_records[-1]['record_hash'][:16]}")
    print(f"contagem prospectiva: {len(realized)}/{PROSPECTIVE_TARGET} semanas liquidadas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
