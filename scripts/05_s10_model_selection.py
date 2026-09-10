"""Leak-safe S10-only selection and final holdout evaluation.

The final holdout never participates in hyperparameter ranking.  The winner is
frozen on expanding validation folds and is then compared with persistence,
Ridge and ARIMA on the last two years.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.production import S10ProductionForecaster
from eval.metrics import diebold_mariano
from vs_epl_krls import load_anp_fuel_csv, regression_report
from vs_epl_krls.selection import (
    build_s10_supervised,
    candidate_grid,
    evaluate_temporal_fold,
    pinned_validation_folds,
    rank_candidates,
)


def _ridge_holdout(data, start: int, end: int, refit_every: int = 13) -> np.ndarray:
    predictions = np.full(end - start, np.nan)
    model = None
    last_fit = -10**9
    for output_index, origin in enumerate(range(start, end)):
        # At origin ``t`` the label created at ``t-h`` has just become known.
        # Include it while keeping every later label unavailable.
        train_end = data.known_target_end(origin)
        if train_end < 40:
            continue
        if model is None or origin - last_fit >= refit_every:
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(data.x[:train_end], data.target_price[:train_end])
            last_fit = origin
        predictions[output_index] = float(model.predict(data.x[origin : origin + 1])[0])
    return predictions


def _arima_holdout(data, start: int, end: int, refit_every: int = 1) -> np.ndarray:
    predictions = np.full(end - start, np.nan)
    model = None
    last_fit = -10**9
    last_history_size = 0
    for output_index, origin in enumerate(range(start, end)):
        history = (
            data.price_history.loc[data.price_history["date"] <= pd.Timestamp(data.dates[origin]), "price"].to_numpy(float)
            if data.price_history is not None else data.origin_price[: origin + 1]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if model is None or origin - last_fit >= refit_every:
                model = S10ProductionForecaster._fit_arima(history)
                last_fit = origin
            else:
                model = model.append(history[last_history_size:], refit=False)
            forecast = np.asarray(model.forecast(steps=data.horizon))
            last_history_size = len(history)
        predictions[output_index] = float(forecast[-1])
    return predictions


def _metric_row(
    name: str,
    actual: np.ndarray,
    prediction: np.ndarray,
    naive: np.ndarray,
    *,
    horizon: int,
) -> dict[str, object]:
    valid = np.isfinite(prediction)
    metrics = regression_report(actual[valid], prediction[valid])
    baseline = regression_report(actual[valid], naive[valid])
    errors = actual[valid] - prediction[valid]
    naive_errors = actual[valid] - naive[valid]
    movement = actual[valid] != naive[valid]
    directional_accuracy = (
        float(
            np.mean(
                np.sign(prediction[valid][movement] - naive[valid][movement])
                == np.sign(actual[valid][movement] - naive[valid][movement])
            )
        )
        if np.any(movement)
        else float("nan")
    )
    dm = diebold_mariano(errors, naive_errors, h=horizon)
    return {
        "model": name,
        **metrics,
        "bias": float(np.mean(prediction[valid] - actual[valid])),
        "max_absolute_error": float(np.max(np.abs(errors))),
        "directional_accuracy": directional_accuracy,
        "rmse_ratio_vs_naive": metrics["rmse"] / baseline["rmse"],
        "dm_vs_naive": dm["dm_stat"],
        "dm_pvalue": dm["pvalue"],
        "n": int(valid.sum()),
    }


def _convex_weights(actual: np.ndarray, predictions: np.ndarray, resolution: int = 20) -> np.ndarray:
    """Non-negative stacking weights fitted by development MAE."""

    if predictions.ndim != 2 or predictions.shape[0] != actual.size:
        raise ValueError("stacking predictions have an incompatible shape")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(actual)):
        raise ValueError("stacking inputs must be finite")
    best_weights = np.zeros(predictions.shape[1], dtype=float)
    best_loss = np.inf

    def partitions(remaining: int, slots: int, prefix: tuple[int, ...] = ()):
        if slots == 1:
            yield prefix + (remaining,)
            return
        for value in range(remaining + 1):
            yield from partitions(remaining - value, slots - 1, prefix + (value,))

    for integer_weights in partitions(resolution, predictions.shape[1]):
        weights = np.asarray(integer_weights, dtype=float) / resolution
        residual = actual - predictions @ weights
        loss = float(np.mean(np.abs(residual)))
        if loss < best_loss:
            best_loss = loss
            best_weights = weights
    return best_weights


def _write_plots(
    output_dir: Path,
    horizon: int,
    ranking: pd.DataFrame,
    predictions: pd.DataFrame,
    interval_lower: np.ndarray,
    interval_upper: np.ndarray,
) -> None:
    dates = pd.to_datetime(predictions["target_date"])
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.fill_between(dates, interval_lower, interval_upper, alpha=0.18, label="P10–P90 ARIMA")
    axis.plot(dates, predictions["actual"], label="real", linewidth=2)
    axis.plot(dates, predictions["arima"], label="ARIMA", linewidth=1.4)
    axis.plot(dates, predictions["vs_epl_krls"], label="VS-ePL-KRLS", alpha=0.85)
    axis.plot(dates, predictions["persistence"], label="persistência", alpha=0.6)
    axis.set(title=f"S10 — holdout final, horizonte {horizon} semana(s)", ylabel="R$/L")
    axis.grid(alpha=0.2)
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(output_dir / f"holdout_forecast_h{horizon}.png", dpi=160)
    plt.close(figure)

    figure, left = plt.subplots(figsize=(13, 4))
    right = left.twinx()
    left.step(dates, predictions["vs_n_rules"], where="post", label="regras", color="tab:blue")
    left.step(
        dates,
        predictions["vs_max_dictionary"],
        where="post",
        label="maior dicionário",
        color="tab:orange",
    )
    right.plot(dates, predictions["vs_beta"], label="beta", color="tab:green", alpha=0.8)
    left.set_ylabel("contagem")
    right.set_ylabel("beta")
    left.grid(alpha=0.2)
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    left.legend(handles_left + handles_right, labels_left + labels_right, ncol=3)
    left.set_title("Evolução estrutural do challenger no holdout")
    figure.tight_layout()
    figure.savefig(output_dir / f"vs_structure_h{horizon}.png", dpi=160)
    plt.close(figure)

    top = ranking.head(15).iloc[::-1]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(top["candidate_id"], top["selection_score"], color="tab:blue", alpha=0.8)
    axis.set(xlabel="score temporal (menor é melhor)", title="Ranking de validação VS-ePL-KRLS")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / f"validation_ranking_h{horizon}.png", dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    s10 = load_anp_fuel_csv(args.data, products=["S10"], weekly="mean")
    datasets = {
        feature_set: build_s10_supervised(s10, horizon=args.horizon, feature_set=feature_set)
        for feature_set in ("price", "lags", "dynamics")
    }
    if len({data.n_samples for data in datasets.values()}) != 1:
        raise RuntimeError("feature sets are not aligned")
    # O corte e resolvido pelas datas-alvo congeladas, nao por ``n_samples``:
    # ancorado no fim da serie, o holdout escorregava uma posicao a cada semana
    # nova da ANP e reabria, sem aviso, uma janela diferente da ja avaliada.
    pinned = pinned_validation_folds(
        datasets["price"].target_dates,
        validation_size=args.validation_size,
        n_folds=args.n_folds,
        expected_holdout_size=args.holdout_size,
    )
    folds, holdout = pinned.folds, pinned.holdout
    if folds[0].validation_start < args.min_train_size:
        raise ValueError("not enough samples for the requested temporal folds")
    n_prospective = pinned.prospective.validation_end - pinned.prospective.validation_start
    candidates = candidate_grid(
        horizon=args.horizon,
        random_state=args.random_state,
        n_random=args.n_random,
        feature_scaling=getattr(args, "feature_scaling", "robust_bounded"),
    )
    digest = hashlib.sha256()
    digest.update(s10.loc[s10["date"] < pd.Timestamp(pinned.holdout_start_date), ["date", "price"]].to_csv(index=False).encode())
    for feature_set, data in datasets.items():
        digest.update(feature_set.encode())
        for values in (data.x, data.target_price, data.dates, data.target_dates):
            digest.update(values[:holdout.validation_start].tobytes())
    digest.update(json.dumps([asdict(c) for c in candidates], sort_keys=True).encode())
    digest.update(json.dumps([asdict(f) for f in folds], sort_keys=True).encode())
    for source in sorted((ROOT / "src" / "vs_epl_krls").glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    digest.update(Path(__file__).read_bytes())
    validation_fingerprint = digest.hexdigest()
    started = time.perf_counter()
    ranking_path = args.output_dir / f"validation_ranking_h{args.horizon}.csv"
    manifest_path = args.output_dir / f"selection_manifest_h{args.horizon}.json"
    previous_manifest: dict[str, object] = {}
    if args.reuse_validation and manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ranking_reused = bool(args.reuse_validation and ranking_path.is_file())
    if ranking_reused:
        if previous_manifest.get("validation_fingerprint") != validation_fingerprint:
            raise ValueError("cached validation has different data, code or configuration; rerun without --reuse-validation")
        ranking = pd.read_csv(ranking_path)
        frozen_id = str(ranking.iloc[0]["candidate_id"])
        frozen_candidate = next(
            candidate for candidate in candidates if candidate.candidate_id == frozen_id
        )
        fold_results = [
            evaluate_temporal_fold(frozen_candidate, datasets[frozen_candidate.feature_set], fold)
            for fold in folds
        ]
    else:
        ranking, fold_results = rank_candidates(candidates, datasets, folds)
    selection_seconds = time.perf_counter() - started
    lookup = {candidate.candidate_id: candidate for candidate in candidates}
    champion = lookup[str(ranking.iloc[0]["candidate_id"])]
    champion_data = datasets[champion.feature_set]
    champion_folds = [
        result for result in fold_results if result.candidate_id == champion.candidate_id
    ]
    validation_actual: list[np.ndarray] = []
    validation_naive: list[np.ndarray] = []
    validation_vs: list[np.ndarray] = []
    validation_ridge: list[np.ndarray] = []
    validation_arima: list[np.ndarray] = []
    for fold, result in zip(folds, champion_folds):
        validation_actual.append(result.actual)
        validation_naive.append(result.naive)
        validation_vs.append(result.predictions)
        validation_ridge.append(
            _ridge_holdout(champion_data, fold.validation_start, fold.validation_end)
        )
        validation_arima.append(
            _arima_holdout(champion_data, fold.validation_start, fold.validation_end)
        )
    oof_actual = np.concatenate(validation_actual)
    oof_matrix = np.column_stack(
        (
            np.concatenate(validation_arima),
            np.concatenate(validation_ridge),
            np.concatenate(validation_naive),
            np.concatenate(validation_vs),
        )
    )
    ensemble_names = ("ARIMA", "Ridge", "persistencia", "VS-ePL-KRLS")
    ensemble_weights = _convex_weights(oof_actual, oof_matrix)
    oof_ensemble = oof_matrix @ ensemble_weights
    validation_comparison = pd.DataFrame(
        [
            _metric_row(name, oof_actual, oof_matrix[:, index], oof_matrix[:, 2], horizon=args.horizon)
            for index, name in enumerate(ensemble_names)
        ]
        + [
            _metric_row("ensemble", oof_actual, oof_ensemble, oof_matrix[:, 2], horizon=args.horizon)
        ]
    ).sort_values("mae", ignore_index=True)

    if not getattr(args, "evaluate_holdout", False):
        ranking.drop(columns=["candidate"], errors="ignore").to_csv(ranking_path, index=False)
        pd.DataFrame([r.summary_row() for r in fold_results]).to_csv(
            args.output_dir / f"validation_folds_h{args.horizon}.csv", index=False
        )
        validation_comparison.to_csv(
            args.output_dir / f"validation_baselines_h{args.horizon}.csv", index=False
        )
        development_manifest = {
            "pipeline_version": "calendar-mae-v2",
            "holdout_evaluated": False,
            "horizon_weeks": args.horizon,
            "validation_fingerprint": validation_fingerprint,
            "validation_folds": [asdict(f) for f in folds],
            "champion_selected_without_holdout": asdict(champion),
            "validation_comparison": validation_comparison.to_dict(orient="records"),
            "selection_metric": "mae_ratio",
            "candidate_search_seconds": selection_seconds,
            "n_candidates_ranked": len(ranking),
            "ensemble_evidence": "weights fitted on these folds; requires independent validation",
        }
        manifest_path.write_text(json.dumps(development_manifest, indent=2, default=str), encoding="utf-8")
        return development_manifest

    holdout_result = evaluate_temporal_fold(champion, champion_data, holdout)

    actual = holdout_result.actual
    naive = holdout_result.naive
    ridge = _ridge_holdout(champion_data, holdout.validation_start, holdout.validation_end)
    arima = _arima_holdout(champion_data, holdout.validation_start, holdout.validation_end)
    holdout_matrix = np.column_stack((arima, ridge, naive, holdout_result.predictions))
    ensemble = holdout_matrix @ ensemble_weights
    comparison = pd.DataFrame(
        [
            _metric_row("VS-ePL-KRLS", actual, holdout_result.predictions, naive, horizon=args.horizon),
            _metric_row("persistencia", actual, naive, naive, horizon=args.horizon),
            _metric_row("Ridge", actual, ridge, naive, horizon=args.horizon),
            _metric_row("ARIMA", actual, arima, naive, horizon=args.horizon),
            _metric_row("ensemble", actual, ensemble, naive, horizon=args.horizon),
        ]
    ).sort_values("rmse", ignore_index=True)

    validation_rows = pd.DataFrame([result.summary_row() for result in fold_results])
    ranking_export = ranking.drop(columns=["candidate"], errors="ignore").copy()
    ranking_export.to_csv(args.output_dir / f"validation_ranking_h{args.horizon}.csv", index=False)
    validation_file = (
        f"champion_validation_folds_h{args.horizon}.csv"
        if args.reuse_validation
        else f"validation_folds_h{args.horizon}.csv"
    )
    validation_rows.to_csv(args.output_dir / validation_file, index=False)
    comparison.to_csv(args.output_dir / f"holdout_comparison_h{args.horizon}.csv", index=False)
    validation_comparison.to_csv(
        args.output_dir / f"validation_baselines_h{args.horizon}.csv",
        index=False,
    )
    holdout_predictions = pd.DataFrame(
        {
            "target_date": holdout_result.dates,
            "actual": actual,
            "vs_epl_krls": holdout_result.predictions,
            "persistence": naive,
            "ridge": ridge,
            "arima": arima,
            "ensemble": ensemble,
            "vs_n_rules": holdout_result.rule_counts,
            "vs_beta": holdout_result.betas,
            "vs_max_dictionary": holdout_result.dictionary_sizes,
        }
    )
    holdout_predictions.to_csv(
        args.output_dir / f"holdout_predictions_h{args.horizon}.csv",
        index=False,
    )

    vs_row = comparison[comparison["model"] == "VS-ePL-KRLS"].iloc[0]
    validation_winner = ranking.iloc[0]
    gates = {
        "validation_beats_naive_all_folds": bool(validation_winner["beats_naive_all_folds"]),
        "holdout_rmse_gain_at_least_2pct": bool(vs_row["rmse_ratio_vs_naive"] <= 0.98),
        "holdout_dm_pvalue_below_005": bool(vs_row["dm_pvalue"] < 0.05),
        "latency_p95_below_20ms": bool(holdout_result.prediction_latency_ms_p95 < 20.0),
        "finite_predictions": bool(np.all(np.isfinite(holdout_result.predictions))),
        "bounded_rules": bool(holdout_result.max_rules <= champion.max_rules),
        "bounded_dictionary": bool(holdout_result.max_dictionary_size <= champion.max_dictionary_size),
    }
    gates["promote_vs_epl_krls"] = bool(all(gates.values()))
    validation_primary = str(validation_comparison.iloc[0]["model"])
    primary_holdout_row = comparison[comparison["model"] == validation_primary].iloc[0]
    production_gates = {
        "selected_on_validation_only": True,
        "finite_holdout_predictions": bool(np.isfinite(primary_holdout_row["rmse"])),
        "holdout_not_worse_than_naive_by_2pct": bool(
            primary_holdout_row["rmse_ratio_vs_naive"] <= 1.02
        ),
        "at_least_100_holdout_predictions": bool(primary_holdout_row["n"] >= 100),
        "holdout_within_2pct_of_arima": bool(
            primary_holdout_row["rmse"]
            <= 1.02 * comparison[comparison["model"] == "ARIMA"].iloc[0]["rmse"]
        ),
    }
    production_gates["production_candidate_passed"] = bool(all(production_gates.values()))
    selected_production_model = (
        validation_primary if production_gates["production_candidate_passed"] else "ARIMA"
    )
    primary_predictions = {
        "ARIMA": arima,
        "Ridge": ridge,
        "persistencia": naive,
        "VS-ePL-KRLS": holdout_result.predictions,
        "ensemble": ensemble,
    }[selected_production_model]
    residual_column = {
        "ARIMA": 0,
        "Ridge": 1,
        "persistencia": 2,
        "VS-ePL-KRLS": 3,
        "ensemble": 4,
    }[selected_production_model]
    oof_residual_matrix = np.column_stack(
        (
            oof_actual - oof_matrix[:, 0],
            oof_actual - oof_matrix[:, 1],
            oof_actual - oof_matrix[:, 2],
            oof_actual - oof_matrix[:, 3],
            oof_actual - oof_ensemble,
        )
    )
    calibration_for_primary = oof_residual_matrix[:, residual_column]
    q10, q90 = np.quantile(calibration_for_primary, [0.10, 0.90])
    interval_lower = primary_predictions + q10
    interval_upper = primary_predictions + q90
    interval_diagnostics = {
        "method": "validation_residual_quantiles",
        "nominal_coverage": 0.80,
        "holdout_coverage": float(
            np.mean((actual >= interval_lower) & (actual <= interval_upper))
        ),
        "mean_width": float(np.mean(interval_upper - interval_lower)),
        "calibration_samples": int(calibration_for_primary.size),
    }
    _write_plots(
        args.output_dir,
        args.horizon,
        ranking,
        holdout_predictions,
        interval_lower,
        interval_upper,
    )
    calibration_residuals = pd.DataFrame(
        {
            "residual_arima": oof_actual - oof_matrix[:, 0],
            "residual_ridge": oof_actual - oof_matrix[:, 1],
            "residual_persistence": oof_actual - oof_matrix[:, 2],
            "residual_vs_epl_krls": oof_actual - oof_matrix[:, 3],
            "residual_ensemble": oof_actual - oof_ensemble,
        }
    )
    calibration_residuals.to_csv(
        args.output_dir / f"calibration_residuals_h{args.horizon}.csv",
        index=False,
    )
    manifest: dict[str, object] = {
        "pipeline_version": "calendar-mae-v2",
        "holdout_evaluated": True,
        "validation_fingerprint": validation_fingerprint,
        "scope": "ANP weekly national resale price — Diesel B S10 only",
        "horizon_weeks": args.horizon,
        "data_start": str(pd.Timestamp(s10["date"].min()).date()),
        "data_end": str(pd.Timestamp(s10["date"].max()).date()),
        "n_observations": int(len(s10)),
        "window_pinned_by": "calendar dates, not sample count",
        "pinned_holdout_dates": {
            "start": pinned.holdout_start_date,
            "end": pinned.holdout_end_date,
        },
        "prospective_samples_outside_protocol": int(n_prospective),
        "validation_folds": [asdict(fold) for fold in folds],
        "holdout": asdict(holdout),
        "champion_selected_without_holdout": asdict(champion),
        "selection_seconds_current_run": selection_seconds,
        "candidate_search_seconds": previous_manifest.get("candidate_search_seconds")
        if ranking_reused
        else selection_seconds,
        "reused_validation_ranking": ranking_reused,
        "n_candidates_ranked": int(len(ranking)),
        "holdout_result": holdout_result.summary_row(),
        "vs_epl_krls_gates": gates,
        "ensemble": {
            "component_order": list(ensemble_names),
            "weights": ensemble_weights.tolist(),
        },
        "validation_comparison": validation_comparison.to_dict(orient="records"),
        "production_gates": production_gates,
        "selected_production_model": selected_production_model,
        "interval_diagnostics": interval_diagnostics,
        "comparison": comparison.to_dict(orient="records"),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "anp_semanal_desde_2013.xlsx")
    parser.add_argument("--horizon", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--n-random", type=int, default=36)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--holdout-size", type=int, default=104)
    parser.add_argument("--validation-size", type=int, default=52)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--min-train-size", type=int, default=156)
    parser.add_argument("--feature-scaling", choices=["minmax", "robust_bounded"], default="robust_bounded")
    parser.add_argument("--evaluate-holdout", action="store_true", help="explicitly evaluate the already-used holdout; default is development only")
    parser.add_argument(
        "--reuse-validation",
        action="store_true",
        help="reusa ranking existente e recalcula apenas o campeão congelado",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "vs_epl_krls" / "s10_selection_v2")
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
