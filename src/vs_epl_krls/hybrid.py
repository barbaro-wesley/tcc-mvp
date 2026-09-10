"""Leak-safe residual hybrid evaluation for one-step S10 forecasting.

The base forecast must be generated causally at every origin.  The evolving
model never sees the price itself as its target: it learns the realized error
``actual - base`` only after that target date has arrived.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import regression_report
from .model import VSEPLKRLS
from .selection import S10Candidate, S10Supervised, TemporalFold


@dataclass
class HybridFoldResult:
    """Result and structural trace of an ARIMA-plus-VS residual fold."""

    candidate_id: str
    fold_id: str
    metrics: dict[str, float]
    base_metrics: dict[str, float]
    naive_metrics: dict[str, float]
    rmse_ratio_vs_base: float
    rmse_ratio_vs_naive: float
    predictions: NDArray[np.float64]
    base_predictions: NDArray[np.float64]
    corrections: NDArray[np.float64]
    actual: NDArray[np.float64]
    dates: NDArray[np.datetime64]
    rule_counts: NDArray[np.int64]
    betas: NDArray[np.float64]
    dictionary_sizes: NDArray[np.int64]
    elapsed_seconds: float
    correction_latency_ms_p95: float
    dictionary_replacements: int
    dictionary_replacement_rate: float
    correction_weight: float
    correction_limit: float | None

    def summary_row(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "fold_id": self.fold_id,
            **self.metrics,
            "base_rmse": self.base_metrics["rmse"],
            "naive_rmse": self.naive_metrics["rmse"],
            "rmse_ratio_vs_base": self.rmse_ratio_vs_base,
            "rmse_ratio_vs_naive": self.rmse_ratio_vs_naive,
            "elapsed_seconds": self.elapsed_seconds,
            "correction_latency_ms_p95": self.correction_latency_ms_p95,
            "max_rules": int(np.max(self.rule_counts, initial=0)),
            "max_dictionary_size": int(np.max(self.dictionary_sizes, initial=0)),
            "dictionary_replacements": self.dictionary_replacements,
            "dictionary_replacement_rate": self.dictionary_replacement_rate,
            "correction_weight": self.correction_weight,
            "correction_limit": self.correction_limit,
        }


def evaluate_residual_hybrid(
    candidate: S10Candidate,
    data: S10Supervised,
    fold: TemporalFold,
    base_predictions: ArrayLike,
) -> HybridFoldResult:
    """Evaluate a causal base forecast corrected by online VS-ePL-KRLS."""

    if candidate.target_mode != "delta":
        raise ValueError("residual hybrid candidates must use target_mode='delta'")
    base = np.asarray(base_predictions, dtype=float).ravel()
    if base.shape != (data.n_samples,):
        raise ValueError("base_predictions must have one value per supervised origin")
    start, end = fold.validation_start, fold.validation_end
    if not data.horizon <= start < end <= data.n_samples:
        raise ValueError("invalid temporal fold bounds")
    if not np.all(np.isfinite(base[start:end])):
        raise ValueError("base_predictions must be finite throughout the evaluation fold")

    x_scaler = candidate.make_feature_scaler().fit(data.x[:start])
    x_scaled = np.clip(x_scaler.transform(data.x), 0.0, 1.0)
    residual_target = data.target_price - base
    model = VSEPLKRLS(**candidate.model_parameters())
    known_end = data.known_target_end(start)
    learned_until = -1
    started = time.perf_counter()
    for index in range(known_end):
        if np.isfinite(residual_target[index]):
            model.learn_one(x_scaled[index], float(residual_target[index]))
        learned_until = index

    corrections: list[float] = []
    latencies: list[float] = []
    rule_counts: list[int] = []
    betas: list[float] = []
    dictionary_sizes: list[int] = []
    for index in range(start, end):
        while learned_until + 1 < data.known_target_end(index):
            newly_available = learned_until + 1
            if np.isfinite(residual_target[newly_available]):
                model.learn_one(
                    x_scaled[newly_available],
                    float(residual_target[newly_available]),
                )
            learned_until = newly_available
        prediction_started = time.perf_counter_ns()
        raw_correction = model.predict_one(x_scaled[index]) if model.n_rules else 0.0
        latencies.append((time.perf_counter_ns() - prediction_started) / 1e6)
        correction = float(candidate.residual_correction_weight * raw_correction)
        if candidate.residual_correction_limit is not None:
            correction = float(
                np.clip(
                    correction,
                    -candidate.residual_correction_limit,
                    candidate.residual_correction_limit,
                )
            )
        corrections.append(float(correction))
        summary = model.summary()
        rule_counts.append(model.n_rules)
        betas.append(model.beta_)
        dictionary_sizes.append(int(summary["max_dictionary_size"]))

    elapsed = time.perf_counter() - started
    correction_array = np.asarray(corrections, dtype=float)
    base_fold = base[start:end]
    prediction = base_fold + correction_array
    actual = data.target_price[start:end]
    naive = data.origin_price[start:end]
    metrics = regression_report(actual, prediction)
    base_metrics = regression_report(actual, base_fold)
    naive_metrics = regression_report(actual, naive)
    final_summary = model.summary()
    return HybridFoldResult(
        candidate_id=candidate.candidate_id,
        fold_id=fold.fold_id,
        metrics=metrics,
        base_metrics=base_metrics,
        naive_metrics=naive_metrics,
        rmse_ratio_vs_base=float(metrics["rmse"] / max(base_metrics["rmse"], 1e-12)),
        rmse_ratio_vs_naive=float(metrics["rmse"] / max(naive_metrics["rmse"], 1e-12)),
        predictions=prediction,
        base_predictions=base_fold.copy(),
        corrections=correction_array,
        actual=actual.copy(),
        dates=data.target_dates[start:end].copy(),
        rule_counts=np.asarray(rule_counts, dtype=np.int64),
        betas=np.asarray(betas, dtype=float),
        dictionary_sizes=np.asarray(dictionary_sizes, dtype=np.int64),
        elapsed_seconds=float(elapsed),
        correction_latency_ms_p95=float(np.quantile(latencies, 0.95)),
        dictionary_replacements=int(final_summary["dictionary_replacements"]),
        dictionary_replacement_rate=float(
            final_summary["dictionary_replacement_rate"]
        ),
        correction_weight=float(candidate.residual_correction_weight),
        correction_limit=candidate.residual_correction_limit,
    )
