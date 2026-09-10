"""B2 -- a decisao de compra em horizontes maiores que uma semana.

Por que este experimento existe
-------------------------------
O produto preve h=1.  Compra de diesel raramente e decidida com sete dias de
antecedencia: contrato, frete e lote minimo empurram a decisao para semanas.  E
ha um argumento empirico do proprio repositorio -- o spread estadual reverte com
meia-vida de ~20 semanas, ou seja, o sinal de medio prazo e mais forte que o de
curto, ao contrario do preco em nivel.

O que ele mede
--------------
Para cada horizonte, replica a **mesma** politica de antecipacao usada em h=1
(``simulate_horizon_prebuy``) sobre previsoes geradas causalmente, e reporta
economia anualizada com IC90 por bootstrap em blocos.  Nada aqui promove modelo:
e um experimento de desenvolvimento.

Como as previsoes sao geradas
-----------------------------
Walk-forward honesto: em cada origem ``t`` o ARIMA e ajustado apenas com o
historico ate ``t`` e projetado ``h`` passos a frente.  A persistencia usa o
preco conhecido em ``t``.  Nenhuma observacao posterior a ``t`` entra na
previsao para ``t+h``.

O holdout congelado NAO e reaberto: por padrao o experimento roda na janela de
desenvolvimento (tudo antes de ``S10_HOLDOUT_START``).  Use ``--window holdout``
apenas se aceitar mais uma leitura do holdout, e saiba que isso e uma decisao de
governanca, nao um detalhe de execucao.

Uso::

    python scripts/33_s10_horizon_experiment.py
    python scripts/33_s10_horizon_experiment.py --horizons 1 2 4 8 12
    python scripts/33_s10_horizon_experiment.py --carrying-cost 0.002
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.fuel import load_anp_fuel_csv  # noqa: E402
from vs_epl_krls.procurement import simulate_horizon_prebuy  # noqa: E402
from vs_epl_krls.selection import S10_HOLDOUT_START  # noqa: E402

DEFAULT_HORIZONS = (1, 2, 4, 8, 12)


def _arima_path(history: np.ndarray, steps: int) -> float:
    """Previsao ARIMA ``steps`` passos a frente, ajustada so com ``history``."""

    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fitted = ARIMA(history, order=(1, 1, 1)).fit()
            return float(np.asarray(fitted.forecast(steps=steps))[-1])
        except Exception:
            # Fallback deliberado: sem convergencia, a persistencia e a
            # previsao honesta, nao um numero inventado por um modelo que falhou.
            return float(history[-1])


def build_horizon_predictions(
    prices: pd.DataFrame,
    *,
    horizon: int,
    start_index: int,
    min_train: int,
) -> pd.DataFrame:
    """Walk-forward causal: origem ``t`` preve ``t+horizon`` sem ver o futuro."""

    dates = pd.to_datetime(prices["date"]).to_numpy()
    values = prices["price"].to_numpy(float)
    rows: list[dict[str, object]] = []
    for origin in range(max(start_index, min_train), len(values) - horizon):
        history = values[: origin + 1]
        target = origin + horizon
        rows.append(
            {
                "target_date": pd.Timestamp(dates[target]),
                "actual": float(values[target]),
                # A origem e o preco conhecido em ``t``: e o que a politica
                # compara contra a previsao, e o que o replay valida.
                "persistence": float(values[origin]),
                "arima": _arima_path(history, horizon),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    fuel = load_anp_fuel_csv(args.data, products=["S10"], weekly="mean")
    prices = (
        fuel.rename(columns={"data": "date", "preco": "price"})
        if "data" in fuel.columns
        else fuel
    )
    prices = prices[["date", "price"]].dropna().reset_index(drop=True)
    prices["date"] = pd.to_datetime(prices["date"])

    holdout_start = pd.Timestamp(S10_HOLDOUT_START)
    development = prices[prices["date"] < holdout_start].reset_index(drop=True)
    if args.window == "development":
        panel, window_label = development, f"desenvolvimento (< {S10_HOLDOUT_START})"
    else:
        panel, window_label = prices, "serie completa (inclui holdout)"

    print(f"janela        : {window_label}")
    print(f"semanas       : {len(panel)}  ({panel['date'].min().date()} a {panel['date'].max().date()})")
    print(f"horizontes    : {list(args.horizons)}")
    print(f"carrego       : R$ {args.carrying_cost:.4f}/L/semana")
    print()

    results: list[dict[str, object]] = []
    for horizon in args.horizons:
        predictions = build_horizon_predictions(
            panel,
            horizon=horizon,
            start_index=args.start_index,
            min_train=args.min_train,
        )
        backtest = simulate_horizon_prebuy(
            predictions,
            horizon=horizon,
            prediction_column="arima",
            model_name=f"ARIMA h={horizon}",
            monthly_liters=args.monthly_liters,
            flexibility_fraction=args.flexibility_fraction,
            signal_threshold_brl_per_liter=args.signal_threshold,
            carrying_cost_brl_per_liter_week=args.carrying_cost,
            random_state=args.random_state,
        )
        errors = np.abs(
            predictions["arima"].to_numpy(float) - predictions["actual"].to_numpy(float)
        )
        naive = np.abs(
            predictions["persistence"].to_numpy(float)
            - predictions["actual"].to_numpy(float)
        )
        low, high = backtest.annualized_savings_ci90_brl
        row = {
            "horizon_weeks": horizon,
            "n_decisions": backtest.n_weeks - horizon,
            "mae": float(np.mean(errors)),
            "persistence_mae": float(np.mean(naive)),
            "mae_ratio_vs_persistence": float(np.mean(errors) / np.mean(naive)),
            "triggered_prebuys": backtest.triggered_prebuys,
            "trigger_precision": backtest.trigger_precision,
            "annualized_savings_brl": backtest.annualized_savings_brl,
            "annualized_savings_ci90_low": low,
            "annualized_savings_ci90_high": high,
            "ci90_positive": bool(low > 0.0),
        }
        results.append(row)
        precision = "n/a" if row["trigger_precision"] is None else f"{row['trigger_precision']:.2f}"
        print(
            f"h={horizon:2d}  decisoes {row['n_decisions']:4d}  "
            f"MAE {row['mae']:.4f} ({row['mae_ratio_vs_persistence']:.2f}x persistencia)  "
            f"disparos {row['triggered_prebuys']:3d} prec {precision}  "
            f"economia/ano R$ {row['annualized_savings_brl']:>10,.0f}  "
            f"IC90 [{low:>9,.0f}, {high:>9,.0f}]"
            f"{'  <-- decidivel' if row['ci90_positive'] else ''}"
        )

    best = max(results, key=lambda r: r["annualized_savings_brl"])
    decidable = [r for r in results if r["ci90_positive"]]
    baseline = next((r for r in results if r["horizon_weeks"] == 1), None)
    print()
    print(f"melhor economia anualizada: h={best['horizon_weeks']}  R$ {best['annualized_savings_brl']:,.0f}")
    if baseline and baseline["annualized_savings_brl"] > 0:
        ratio = best["annualized_savings_brl"] / baseline["annualized_savings_brl"]
        print(f"contra h=1               : {ratio:.2f}x")
    print(
        "horizontes com IC90 positivo: "
        + (", ".join(f"h={r['horizon_weeks']}" for r in decidable) if decidable else "nenhum")
    )

    payload: dict[str, object] = {
        "experiment": "B2_horizonte_maior",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window": args.window,
        "window_label": window_label,
        "holdout_reopened": args.window != "development",
        "n_weeks": int(len(panel)),
        "period_start": str(panel["date"].min().date()),
        "period_end": str(panel["date"].max().date()),
        "policy": {
            "monthly_liters": args.monthly_liters,
            "flexibility_fraction": args.flexibility_fraction,
            "signal_threshold_brl_per_liter": args.signal_threshold,
            "carrying_cost_brl_per_liter_week": args.carrying_cost,
            "carrying_cost_scales_with_horizon": True,
        },
        "results": results,
        "best_horizon": best["horizon_weeks"],
        "decidable_horizons": [r["horizon_weeks"] for r in decidable],
        "claim_boundary": (
            "replay historico de politica em janela de desenvolvimento; nao promove "
            "modelo, nao reabre o holdout congelado e nao garante economia futura"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "horizon_experiment.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pd.DataFrame(results).to_csv(args.output_dir / "horizon_comparison.csv", index=False)
    print(f"\ngravado em {args.output_dir}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "raw" / "anp_semanal_desde_2013.xlsx"
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument(
        "--window",
        choices=["development", "holdout"],
        default="development",
        help="development nao le o holdout congelado; holdout le (decisao de governanca)",
    )
    parser.add_argument("--monthly-liters", type=float, default=200_000.0)
    parser.add_argument("--flexibility-fraction", type=float, default=0.25)
    parser.add_argument("--signal-threshold", type=float, default=0.01)
    parser.add_argument("--carrying-cost", type=float, default=0.0)
    parser.add_argument("--min-train", type=int, default=156)
    parser.add_argument("--start-index", type=int, default=400)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_horizon",
    )
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
