"""Research-only S10 next-challenger selection without reopening the holdout.

This script compares targeted VS-ePL-KRLS stability extensions and a causal
ARIMA-plus-VS residual hybrid on the already designated development folds.  It
never evaluates a new candidate on the final 104-week holdout, which has
already been opened and cannot provide fresh promotion evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.classical import arima_forecast
from vs_epl_krls.hybrid import HybridFoldResult, evaluate_residual_hybrid
from vs_epl_krls.selection import (
    S10Candidate,
    S10Supervised,
    TemporalFold,
    build_s10_supervised,
    pinned_validation_folds,
    rank_candidates,
)


def _load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "data",
        "revenda",
        "brent_l1",
        "usdbrl_l1",
        "brent_brl_l1",
        "petrobras_reajuste_l1",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"causal S10 panel is missing columns: {sorted(missing)}")
    return frame.rename(columns={"data": "date", "revenda": "price"})


def _causal_arima_predictions(
    data: S10Supervised,
    *,
    end: int,
    min_history: int = 52,
    refit_every: int = 13,
) -> np.ndarray:
    """Generate one-step forecasts using only prices known at each origin."""

    if data.horizon != 1:
        raise ValueError("the residual hybrid currently supports horizon=1 only")
    if not min_history <= end <= data.n_samples:
        raise ValueError("invalid causal ARIMA range")
    predictions = np.full(data.n_samples, np.nan, dtype=float)
    model = None
    last_fit = -10**9
    last_history_size = 0
    for origin in range(min_history, end):
        calendar = getattr(data, "price_history", None)
        history = (
            calendar.loc[calendar["date"] <= pd.Timestamp(data.dates[origin]), "price"].to_numpy(float)
            if calendar is not None else data.origin_price[: origin + 1]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model is None or origin - last_fit >= refit_every or len(history) - last_history_size != 1:
                forecast, model = arima_forecast(history, steps=1, model=None)
                last_fit = origin
            else:
                forecast, model = arima_forecast(history, steps=1, model=model)
        last_history_size = len(history)
        predictions[origin] = float(forecast[0])
    return predictions


def _targeted_direct_candidates(frozen: S10Candidate) -> list[S10Candidate]:
    candidates = [replace(frozen, candidate_id="current_vs_baseline")]
    for rate in (0.10, 0.25, 0.50):
        candidates.append(
            replace(
                frozen,
                candidate_id=f"vs_beta_recovery_{rate:.2f}",
                beta_recovery_rate=rate,
            )
        )
    for beta_min in (0.001, 0.005):
        candidates.append(
            replace(
                frozen,
                candidate_id=f"vs_beta_floor_{beta_min:g}",
                beta_min=beta_min,
                beta_recovery_rate=0.25,
            )
        )
    for forgetting in (0.995, 0.98):
        candidates.append(
            replace(
                frozen,
                candidate_id=f"vs_forgetting_{forgetting:g}",
                forgetting_factor=forgetting,
                dictionary_usage_decay=0.995,
            )
        )
    for size in (30, 40):
        candidates.append(
            replace(
                frozen,
                candidate_id=f"vs_dictionary_{size}",
                max_dictionary_size=size,
                dictionary_usage_decay=0.99,
            )
        )
    candidates.extend(
        [
            replace(
                frozen,
                candidate_id="vs_exogenous_paper",
                feature_set="exogenous",
            ),
            replace(
                frozen,
                candidate_id="vs_exogenous_adaptive",
                feature_set="exogenous",
                beta_min=0.001,
                beta_recovery_rate=0.25,
                forgetting_factor=0.995,
                dictionary_usage_decay=0.99,
                max_dictionary_size=30,
            ),
            replace(
                frozen,
                candidate_id="vs_dynamics_adaptive",
                feature_set="dynamics",
                beta_min=0.001,
                beta_recovery_rate=0.25,
                forgetting_factor=0.995,
                dictionary_usage_decay=0.99,
                max_dictionary_size=30,
            ),
        ]
    )
    return candidates


def _targeted_hybrid_candidates(frozen: S10Candidate) -> list[S10Candidate]:
    candidates: list[S10Candidate] = []
    for feature_set in ("lags", "dynamics", "exogenous"):
        candidates.extend(
            [
                replace(
                    frozen,
                    candidate_id=f"hybrid_{feature_set}_paper",
                    feature_set=feature_set,  # type: ignore[arg-type]
                    target_mode="delta",
                ),
                replace(
                    frozen,
                    candidate_id=f"hybrid_{feature_set}_conservative",
                    feature_set=feature_set,  # type: ignore[arg-type]
                    target_mode="delta",
                    beta_min=0.001,
                    beta_recovery_rate=0.10,
                    dictionary_usage_decay=0.995,
                    residual_correction_weight=0.50,
                    residual_correction_limit=0.10,
                ),
                replace(
                    frozen,
                    candidate_id=f"hybrid_{feature_set}_adaptive",
                    feature_set=feature_set,  # type: ignore[arg-type]
                    target_mode="delta",
                    beta_min=0.001,
                    beta_recovery_rate=0.25,
                    forgetting_factor=0.995,
                    dictionary_usage_decay=0.99,
                    max_dictionary_size=30,
                    residual_correction_weight=0.75,
                    residual_correction_limit=0.15,
                ),
            ]
        )
    return candidates


def _rank_hybrids(
    candidates: list[S10Candidate],
    datasets: dict[str, S10Supervised],
    folds: list[TemporalFold],
    base_predictions: np.ndarray,
) -> tuple[pd.DataFrame, list[HybridFoldResult]]:
    results: list[HybridFoldResult] = []
    lookup = {candidate.candidate_id: candidate for candidate in candidates}
    for candidate in candidates:
        for fold in folds:
            results.append(
                evaluate_residual_hybrid(
                    candidate,
                    datasets[candidate.feature_set],
                    fold,
                    base_predictions,
                )
            )
    rows = pd.DataFrame([result.summary_row() for result in results])
    ranking = rows.groupby("candidate_id", as_index=False).agg(
        mean_rmse=("rmse", "mean"),
        worst_rmse=("rmse", "max"),
        mean_rmse_ratio_vs_base=("rmse_ratio_vs_base", "mean"),
        worst_rmse_ratio_vs_base=("rmse_ratio_vs_base", "max"),
        mean_rmse_ratio_vs_naive=("rmse_ratio_vs_naive", "mean"),
        ratio_std=("rmse_ratio_vs_base", "std"),
        latency_ms_p95=("correction_latency_ms_p95", "max"),
        max_rules=("max_rules", "max"),
        max_dictionary_size=("max_dictionary_size", "max"),
        replacement_rate=("dictionary_replacement_rate", "max"),
    )
    ranking["selection_score"] = (
        ranking["mean_rmse_ratio_vs_base"]
        + 0.20 * ranking["worst_rmse_ratio_vs_base"]
        + 0.10 * ranking["ratio_std"].fillna(0.0)
    )
    ranking["beats_arima_all_folds"] = ranking["worst_rmse_ratio_vs_base"] < 1.0
    ranking["bounded_replacement_churn"] = ranking["replacement_rate"] <= 0.40
    ranking["eligible_for_shadow"] = (
        ranking["beats_arima_all_folds"] & ranking["bounded_replacement_churn"]
    )
    ranking["candidate"] = ranking["candidate_id"].map(
        lambda identifier: asdict(lookup[str(identifier)])
    )
    return ranking.sort_values(
        ["eligible_for_shadow", "selection_score", "mean_rmse", "latency_ms_p95"],
        ascending=[False, True, True, True],
        ignore_index=True,
    ), results


def _development_comparison(
    direct_results,
    hybrid_results: list[HybridFoldResult],
    direct_winner: str,
    hybrid_winner: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected_direct = [r for r in direct_results if r.candidate_id == direct_winner]
    selected_hybrid = [r for r in hybrid_results if r.candidate_id == hybrid_winner]
    baseline_direct = [r for r in direct_results if r.candidate_id == "current_vs_baseline"]
    for name, results, metric in (
        ("VS atual", baseline_direct, "metrics"),
        ("VS next", selected_direct, "metrics"),
        ("ARIMA+VS residual", selected_hybrid, "metrics"),
        ("ARIMA", selected_hybrid, "base_metrics"),
        ("persistência", selected_hybrid, "naive_metrics"),
    ):
        metric_rows = [getattr(result, metric) if hasattr(result, metric) else result.metrics for result in results]
        rows.append(
            {
                "model": name,
                "mean_rmse": float(np.mean([row["rmse"] for row in metric_rows])),
                "worst_rmse": float(np.max([row["rmse"] for row in metric_rows])),
                "mean_mae": float(np.mean([row["mae"] for row in metric_rows])),
                "mean_smape": float(np.mean([row["smape"] for row in metric_rows])),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_rmse", ignore_index=True)


def _plots(output_dir: Path, direct: pd.DataFrame, hybrid: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, ranking, title in (
        (axes[0], direct, "VS direto"),
        (axes[1], hybrid, "ARIMA + VS residual"),
    ):
        top = ranking.head(10).iloc[::-1]
        axis.barh(top["candidate_id"], top["selection_score"], alpha=0.8)
        axis.set_title(title)
        axis.set_xlabel("score temporal (menor é melhor)")
        axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "next_challenger_ranking.png", dpi=160)
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    production_manifest = json.loads(args.production_manifest.read_text(encoding="utf-8"))
    frozen = S10Candidate(**production_manifest["champion_selected_without_holdout"])
    folds = [TemporalFold(**values) for values in production_manifest["validation_folds"]]
    development_end = max(fold.validation_end for fold in folds)
    panel = _load_panel(args.panel)
    datasets = {
        feature_set: build_s10_supervised(panel, horizon=1, feature_set=feature_set)
        for feature_set in ("lags", "dynamics", "exogenous")
    }
    if len({data.n_samples for data in datasets.values()}) != 1:
        raise RuntimeError("next-gen feature sets are not origin-aligned")
    # Os indices herdados do manifesto so continuam apontando para as mesmas
    # semanas enquanto o alinhamento do painel nao muda.  Resolver a janela
    # congelada pelas datas e comparar transforma qualquer deslocamento
    # silencioso em falha explicita.
    pinned = pinned_validation_folds(datasets["lags"].target_dates)
    if [asdict(fold) for fold in pinned.folds] != [asdict(fold) for fold in folds]:
        raise RuntimeError(
            "os folds do manifesto de producao nao correspondem mais as datas "
            f"congeladas {pinned.holdout_start_date}..{pinned.holdout_end_date}; "
            "reavalie o protocolo antes de comparar challengers"
        )

    started = time.perf_counter()
    base = _causal_arima_predictions(datasets["lags"], end=development_end)
    direct_candidates = _targeted_direct_candidates(frozen)
    direct_ranking, direct_results = rank_candidates(
        direct_candidates,
        datasets,  # type: ignore[arg-type]
        folds,
    )
    hybrid_candidates = _targeted_hybrid_candidates(frozen)
    hybrid_ranking, hybrid_results = _rank_hybrids(
        hybrid_candidates,
        datasets,
        folds,
        base,
    )
    elapsed = time.perf_counter() - started
    direct_winner = str(direct_ranking.iloc[0]["candidate_id"])
    hybrid_winner = str(hybrid_ranking.iloc[0]["candidate_id"])
    best_average_hybrid = str(
        hybrid_ranking.sort_values("mean_rmse_ratio_vs_base").iloc[0]["candidate_id"]
    )
    comparison = _development_comparison(
        direct_results,
        hybrid_results,
        direct_winner,
        hybrid_winner,
    )

    direct_ranking.drop(columns=["candidate"]).to_csv(
        args.output_dir / "direct_ranking.csv", index=False
    )
    hybrid_ranking.drop(columns=["candidate"]).to_csv(
        args.output_dir / "hybrid_ranking.csv", index=False
    )
    pd.DataFrame([result.summary_row() for result in direct_results]).to_csv(
        args.output_dir / "direct_folds.csv", index=False
    )
    pd.DataFrame([result.summary_row() for result in hybrid_results]).to_csv(
        args.output_dir / "hybrid_folds.csv", index=False
    )
    comparison.to_csv(args.output_dir / "development_comparison.csv", index=False)
    _plots(args.output_dir, direct_ranking, hybrid_ranking)

    direct_candidate = next(
        candidate for candidate in direct_candidates if candidate.candidate_id == direct_winner
    )
    hybrid_candidate = next(
        candidate for candidate in hybrid_candidates if candidate.candidate_id == hybrid_winner
    )
    payload: dict[str, object] = {
        "scope": "S10 next challenger — development folds only",
        "holdout_evaluated": False,
        "production_promotion_allowed": False,
        "reason": "the existing holdout has already been opened; fresh future data is required",
        "development_end_index": development_end,
        "folds": [asdict(fold) for fold in folds],
        "elapsed_seconds": elapsed,
        "direct_candidates": len(direct_candidates),
        "hybrid_candidates": len(hybrid_candidates),
        "best_direct": asdict(direct_candidate),
        "best_hybrid": asdict(hybrid_candidate),
        "best_average_hybrid": best_average_hybrid,
        "best_direct_validation": direct_ranking.iloc[0].drop("candidate").to_dict(),
        "best_hybrid_validation": hybrid_ranking.iloc[0].drop("candidate").to_dict(),
        "comparison": comparison.to_dict(orient="records"),
    }
    (args.output_dir / "next_challenger_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    report = [
        "# S10 next challenger — relatório de desenvolvimento",
        "",
        "Este experimento não reabriu o holdout final e não autoriza promoção.",
        "",
        f"- Candidatos VS diretos: {len(direct_candidates)}",
        f"- Candidatos híbridos: {len(hybrid_candidates)}",
        f"- Tempo: {elapsed:.2f} s",
        f"- Melhor VS direto: `{direct_winner}`",
        f"- Melhor híbrido: `{hybrid_winner}`",
        f"- Melhor média híbrida (pode ser instável): `{best_average_hybrid}`",
        "",
        "## Comparação média nos folds de desenvolvimento",
        "",
        _markdown_table(comparison),
        "",
        "## Decisão",
        "",
        "O candidato elegível fica em shadow research. Para ser elegível, precisa superar ARIMA em todos os folds e manter churn de substituição em até 40%. Somente dados futuros ainda não observados podem liberar promoção.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=ROOT / "data" / "processed" / "semanal_s10_features.csv",
    )
    parser.add_argument(
        "--production-manifest",
        type=Path,
        default=ROOT
        / "reports"
        / "vs_epl_krls"
        / "s10_selection"
        / "selection_manifest_h1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_next",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
