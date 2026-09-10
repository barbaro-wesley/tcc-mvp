"""Read-only training diagnostics; new predictions use development data only."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.classical import arima_forecast
from vs_epl_krls.production import S10ProductionForecaster
from vs_epl_krls.passthrough import PARITY_FEATURES, PassThroughECM, build_parity_panel
from vs_epl_krls.selection import (
    S10_HOLDOUT_START, build_s10_supervised, pinned_validation_folds,
)
from vs_epl_krls.utils import MinMaxScaler


def run():
    panel = pd.read_csv(ROOT / "data/processed/s10_causal_panel.csv", parse_dates=["date"])
    legacy = pd.read_csv(ROOT / "data/processed/semanal_s10_features.csv")
    supervised = build_s10_supervised(panel, feature_set="lags")
    windows = pinned_validation_folds(supervised.target_dates)
    days = (supervised.target_dates - supervised.dates) / np.timedelta64(1, "D")
    bad = np.flatnonzero(days != 7)
    result = {
        "scope": "diagnostics only; no promotion, no new holdout predictions",
        "runtime": {"numpy": np.__version__, "pandas": pd.__version__},
        "panel_rows": len(panel),
        "legacy_missing_fraction": {c: float(legacy[c].isna().mean()) for c in
                                    ("ulsd", "ulsd_l1", "distribuicao_l1")},
        "causal_missing_fraction": {c: float(panel[c].isna().mean()) for c in
                                    ("ulsd", "parity", "producer_price")},
        "nonweekly_h1_pairs": [
            {"origin": str(supervised.dates[i])[:10],
             "target": str(supervised.target_dates[i])[:10], "days": float(days[i])}
            for i in bad
        ],
    }
    clips = []
    for fold in windows.folds:
        scaler = MinMaxScaler().fit(supervised.x[:fold.validation_start])
        transformed = scaler.transform(supervised.x[fold.validation_start:fold.validation_end])
        outside = (transformed < 0) | (transformed > 1)
        clips.append({"fold": fold.fold_id, "cell_fraction": float(outside.mean()),
                      "rows_with_clipping_fraction": float(outside.any(axis=1).mean()),
                      "rows": len(transformed),
                      "unique_raw_rows": len(np.unique(supervised.x[fold.validation_start:fold.validation_end], axis=0)),
                      "unique_clipped_rows": len(np.unique(np.clip(transformed, 0, 1), axis=0)),
                      "start_target": str(supervised.target_dates[fold.validation_start])[:10],
                      "end_target": str(supervised.target_dates[fold.validation_end-1])[:10]})
    result["lags_scaler_development_clipping"] = clips

    existing = pd.read_csv(ROOT / "reports/vs_epl_krls/s10_parity/holdout_predictions.csv")
    actual, naive = existing.actual.to_numpy(), existing.persistence.to_numpy()
    quiet = np.abs(actual - naive) <= 0.02
    metrics = {}
    for name in ("arima", "paridade", "persistence"):
        pred = existing[name].to_numpy()
        error = actual - pred
        se = np.sort(error**2)[::-1]
        moving = actual != naive
        metrics[name] = {
            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error**2).mean())),
            "within_1cent_fraction": float((np.abs(error) <= 0.01 + 1e-12).mean()),
            "within_2cents_fraction": float((np.abs(error) <= 0.02 + 1e-12).mean()),
            "quiet_mae_threshold_002": float(np.abs(error[quiet]).mean()),
            "direction_moving_only": float((np.sign(pred[moving]-naive[moving]) == np.sign(actual[moving]-naive[moving])).mean()),
            "moving_weeks": int(moving.sum()), "all_weeks": len(actual),
            "top1_squared_error_share": float(se[:1].sum()/se.sum()),
            "top3_squared_error_share": float(se[:3].sum()/se.sum()),
        }
    result["recomputed_published_holdout_predictions"] = metrics
    # The two implementations use the same history here to isolate refit/grid differences.
    development = panel.loc[panel.date < pd.Timestamp(S10_HOLDOUT_START), ["date", "price"]].reset_index(drop=True)
    causal_dev = panel.loc[panel.date < pd.Timestamp(S10_HOLDOUT_START) - pd.Timedelta(days=7)].reset_index(drop=True)
    parity_dev = build_parity_panel(causal_dev)
    last = parity_dev.iloc[-1]
    origin_price = float(last.price)
    delta = np.log(causal_dev.parity).diff()
    emitted = pd.Series({
        "dp1": float(last.y), "rpar1": float(delta.iloc[-1] * origin_price),
        "rpar2": float(delta.iloc[-2] * origin_price),
        "coint_par": float(last.coint_par / last.origin_price * origin_price),
        "volatility": float(last.volatility),
        "abs_cost_move": abs(float(delta.iloc[-1] * origin_price)),
    })
    next_features = []
    for multiplier in (0.9, 1.1):
        placeholder = causal_dev.iloc[[-1]].copy()
        placeholder["date"] += pd.Timedelta(days=7)
        placeholder["price"] *= multiplier
        rebuilt = build_parity_panel(pd.concat([causal_dev, placeholder], ignore_index=True))
        next_features.append(rebuilt.iloc[-1][emitted.index].astype(float))
    assert np.allclose(next_features[0], next_features[1])
    fitted_parity = PassThroughECM(feature_names=PARITY_FEATURES).fit(parity_dev)
    emitted_point = fitted_parity.forecast_row(emitted, origin_price=origin_price).point
    canonical_point = fitted_parity.forecast_row(next_features[0], origin_price=origin_price).point
    result["parity_next_feature_consistency_development"] = {
        "origin_date": str(causal_dev.date.iloc[-1].date()),
        "placeholder_target_invariance_verified": True,
        "manual_serving_features": emitted.to_dict(),
        "canonical_next_features": next_features[0].to_dict(),
        "manual_point": emitted_point, "canonical_point": canonical_point,
        "point_difference_brl_per_liter": canonical_point - emitted_point,
    }
    rows = []
    state = None
    start = len(development) - 27
    for origin in range(start, len(development)-1):
        history = development.price.to_numpy()[:origin+1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if (origin-start) % 13 == 0:
                state = None
            prediction, state = arima_forecast(history, model=state)
            live = S10ProductionForecaster._fit_arima(history)
            production_prediction = float(np.asarray(live.forecast(1))[0])
        rows.append({"target_date": str(development.date.iloc[origin+1].date()),
                     "actual": float(development.price.iloc[origin+1]),
                     "selection_arima": float(prediction[0]),
                     "production_arima": production_prediction,
                     "selection_order": list(state.model.order),
                     "production_order": list(live.model.order)})
    values = pd.DataFrame(rows)
    result["arima_development_comparison"] = {
        "scope": "26 final development targets, common full history, grid and refit only; excludes bundle fallback",
        "start": rows[0]["target_date"], "end": rows[-1]["target_date"],
        "mean_absolute_prediction_difference": float((values.selection_arima-values.production_arima).abs().mean()),
        "max_absolute_prediction_difference": float((values.selection_arima-values.production_arima).abs().max()),
        "selection_mae": float((values.actual-values.selection_arima).abs().mean()),
        "production_mae": float((values.actual-values.production_arima).abs().mean()),
        "different_order_count": sum(r["selection_order"] != r["production_order"] for r in rows),
        "rows": rows,
    }
    out = Path(__file__).with_name("diagnostics.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["arima_development_comparison"].pop("rows")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
