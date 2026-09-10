"""Prospective shadow evaluation for the frozen S10 residual hybrid.

This module is separate from the production bundle. It freezes preprocessing
and parameters at a cutoff, issues one forecast, and learns its residual only
after the corresponding weekly observation is supplied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import erfc, sqrt
from pathlib import Path
import platform
import threading
from typing import Any, Literal
import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.arima.model import ARIMA

from .metrics import regression_report
from .model import VSEPLKRLS
from .selection import S10Candidate, build_s10_feature_frame, build_s10_supervised
from .weekly import weekly_grid


@dataclass(frozen=True)
class S10ShadowForecast:
    """One frozen, prospective ARIMA-plus-VS forecast."""

    forecast_id: str
    target_date: str
    base_point: float
    raw_correction: float
    applied_correction: float
    point: float
    persistence: float
    guardrail_used: bool
    last_observed_date: str
    last_observed_price: float
    candidate_fingerprint: str
    data_fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class S10ShadowHealth:
    """Operational state of the prospective challenger."""

    status: Literal["healthy", "warning"]
    n_observations: int
    n_outcomes: int
    pending_target_date: str | None
    n_rules: int
    beta: float
    max_dictionary_size: int
    dictionary_replacements: int
    dictionary_replacement_rate: float
    feature_clip_fraction: float
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class S10ResidualHybridShadow:
    """Serializable prospective monitor for a frozen residual hybrid."""

    artifact_version = "1.0.0"

    def __init__(
        self,
        candidate: S10Candidate,
        *,
        expected_frequency_days: int = 7,
        min_arima_history: int = 52,
        arima_refit_every: int = 13,
    ) -> None:
        if candidate.target_mode != "delta":
            raise ValueError("shadow residual candidate must use target_mode='delta'")
        if candidate.feature_set not in {"price", "lags", "dynamics"}:
            raise ValueError("shadow artifact does not ingest exogenous features")
        if expected_frequency_days < 1 or min_arima_history < 20 or arima_refit_every < 1:
            raise ValueError("invalid shadow timing configuration")
        self.candidate = candidate
        self.expected_frequency_days = int(expected_frequency_days)
        self.min_arima_history = int(min_arima_history)
        self.arima_refit_every = int(arima_refit_every)
        self.artifact_version_ = self.artifact_version
        self.created_at_utc_ = datetime.now(timezone.utc).isoformat()
        self.candidate_fingerprint_ = self._json_hash(asdict(candidate))
        self._lock = threading.RLock()
        self.is_fitted_ = False

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    @staticmethod
    def _json_hash(value: Any) -> str:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _validate_history(frame: pd.DataFrame) -> pd.DataFrame:
        if not {"date", "price"}.issubset(frame.columns):
            raise ValueError("history must contain date and price")
        history = frame[["date", "price"]].copy()
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
        history["price"] = pd.to_numeric(history["price"], errors="coerce")
        if history.isna().any().any() or (history["price"] <= 0).any():
            raise ValueError("history contains invalid dates or prices")
        if history["date"].duplicated().any() or not history["date"].is_monotonic_increasing:
            raise ValueError("history dates must be unique and strictly increasing")
        if len(history) < 80:
            raise ValueError("at least 80 weekly S10 observations are required")
        return history.reset_index(drop=True)

    @staticmethod
    def _data_fingerprint(history: pd.DataFrame) -> str:
        values = pd.util.hash_pandas_object(
            history[["date", "price"]], index=False
        ).to_numpy()
        return hashlib.sha256(values.tobytes()).hexdigest()

    @staticmethod
    def _fit_arima(prices: np.ndarray):
        best = None
        best_aic = np.inf
        for order in ((0, 1, 1), (1, 1, 0), (1, 1, 1)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = ARIMA(prices, order=order, trend="n").fit(
                        method_kwargs={"warn_convergence": False}
                    )
                if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                    best, best_aic = fitted, float(fitted.aic)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
        if best is None:
            raise RuntimeError("all ARIMA candidates failed")
        return best

    def _causal_base_predictions(self, supervised) -> np.ndarray:
        calendar = weekly_grid(self.history_)
        prices = calendar["price"].to_numpy(float)
        history_dates = calendar["date"].to_numpy(dtype="datetime64[ns]")
        output = np.full(supervised.n_samples, np.nan, dtype=float)
        fitted = None
        last_fit_position = -10**9
        previous_position = -1
        for index, origin_date in enumerate(supervised.dates):
            position = int(np.searchsorted(history_dates, origin_date, side="right") - 1)
            if position + 1 < self.min_arima_history:
                continue
            history = prices[: position + 1]
            try:
                if fitted is None or position - last_fit_position >= self.arima_refit_every:
                    fitted = self._fit_arima(history)
                    last_fit_position = position
                else:
                    fitted = fitted.append(prices[previous_position + 1 : position + 1], refit=False)
                output[index] = float(np.asarray(fitted.forecast(steps=1)).ravel()[0])
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                fitted = self._fit_arima(history)
                last_fit_position = position
                output[index] = float(np.asarray(fitted.forecast(steps=1)).ravel()[0])
            previous_position = position
        return output

    def fit(self, history: pd.DataFrame) -> "S10ResidualHybridShadow":
        """Freeze preprocessing and train residual state on cutoff data."""

        with self._lock:
            self.history_ = self._validate_history(history)
            supervised = build_s10_supervised(
                self.history_, horizon=1, feature_set=self.candidate.feature_set
            )
            self.feature_names_ = supervised.feature_names
            self.x_scaler_ = self.candidate.make_feature_scaler().fit(supervised.x)
            x_scaled = np.clip(self.x_scaler_.transform(supervised.x), 0.0, 1.0)
            base = self._causal_base_predictions(supervised)
            self.vs_model_ = VSEPLKRLS(**self.candidate.model_parameters())
            learned = 0
            for index in range(supervised.n_samples):
                if np.isfinite(base[index]):
                    residual = float(supervised.target_price[index] - base[index])
                    self.vs_model_.learn_one(x_scaled[index], residual)
                    learned += 1
            if learned < 20:
                raise RuntimeError("insufficient causal residuals to fit shadow model")
            self.baseline_training_samples_ = learned
            self.arima_model_ = self._fit_arima(
                weekly_grid(self.history_)["price"].to_numpy(float)
            )
            self.data_fingerprint_ = self._data_fingerprint(self.history_)
            self.freeze_start_ = str(self.history_["date"].iloc[0].date())
            self.freeze_cutoff_ = str(self.history_["date"].iloc[-1].date())
            self.pending_forecast_: S10ShadowForecast | None = None
            self.pending_scaled_features_: np.ndarray | None = None
            self.evaluations_: list[dict[str, object]] = []
            self.last_feature_clip_fraction_ = 0.0
            self.is_fitted_ = True
            return self

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("shadow forecaster has not been fitted")

    def _latest_features(self) -> tuple[np.ndarray, float]:
        feature_frame = build_s10_feature_frame(
            self.history_, feature_set=self.candidate.feature_set
        )
        if feature_frame.empty or feature_frame["date"].iloc[-1] != self.history_["date"].iloc[-1]:
            raise ValueError("latest week needs 13 consecutive observed weeks of features")
        raw = feature_frame[list(self.feature_names_)].iloc[-1].to_numpy(float)
        transformed = self.x_scaler_.transform(raw.reshape(1, -1))[0]
        scaled = np.clip(transformed, 0.0, 1.0)
        clipped = float(np.mean(np.abs(transformed - scaled) > 1e-12))
        return scaled, clipped

    def predict_next(self) -> S10ShadowForecast:
        """Calculate the next shadow forecast without changing state."""

        with self._lock:
            self._require_fitted()
            scaled, clipped = self._latest_features()
            self.last_feature_clip_fraction_ = clipped
            last_price = float(self.history_["price"].iloc[-1])
            base = float(np.asarray(self.arima_model_.forecast(steps=1)).ravel()[0])
            guardrail = False
            if not np.isfinite(base):
                base, guardrail = last_price, True
            raw = float(self.vs_model_.predict_one(scaled))
            if not np.isfinite(raw):
                raw, guardrail = 0.0, True
            applied = float(self.candidate.residual_correction_weight * raw)
            if self.candidate.residual_correction_limit is not None:
                bounded = float(
                    np.clip(
                        applied,
                        -self.candidate.residual_correction_limit,
                        self.candidate.residual_correction_limit,
                    )
                )
                guardrail = guardrail or not np.isclose(bounded, applied)
                applied = bounded
            point = base + applied
            if not np.isfinite(point) or point <= 0:
                point, applied, guardrail = base, 0.0, True
            last_date = pd.Timestamp(self.history_["date"].iloc[-1])
            target_date = last_date + pd.Timedelta(days=self.expected_frequency_days)
            forecast_id = self._json_hash(
                {
                    "candidate": self.candidate_fingerprint_,
                    "data": self.data_fingerprint_,
                    "target_date": str(target_date.date()),
                }
            )
            return S10ShadowForecast(
                forecast_id=forecast_id,
                target_date=str(target_date.date()),
                base_point=float(base),
                raw_correction=raw,
                applied_correction=applied,
                point=float(point),
                persistence=last_price,
                guardrail_used=guardrail,
                last_observed_date=str(last_date.date()),
                last_observed_price=last_price,
                candidate_fingerprint=self.candidate_fingerprint_,
                data_fingerprint=self.data_fingerprint_,
            )

    def issue_forecast(self) -> S10ShadowForecast:
        """Freeze the next forecast so its outcome can later be matched exactly."""

        with self._lock:
            self._require_fitted()
            if self.pending_forecast_ is not None:
                return self.pending_forecast_
            scaled, _ = self._latest_features()
            forecast = self.predict_next()
            self.pending_scaled_features_ = scaled.copy()
            self.pending_forecast_ = forecast
            return forecast

    def update_one(self, date: str | pd.Timestamp, price: float) -> dict[str, object]:
        """Score the issued forecast, then reveal its residual to the VS model."""

        with self._lock:
            self._require_fitted()
            if self.pending_forecast_ is None or self.pending_scaled_features_ is None:
                raise RuntimeError("issue_forecast must be called before update_one")
            timestamp = pd.Timestamp(date)
            if str(timestamp.date()) != self.pending_forecast_.target_date:
                raise ValueError("outcome date must match the pending target_date")
            value = float(price)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("price must be finite and positive")
            pending = self.pending_forecast_
            residual_target = value - pending.base_point
            self.vs_model_.learn_one(self.pending_scaled_features_, residual_target)
            evaluation: dict[str, object] = {
                "forecast_id": pending.forecast_id,
                "target_date": pending.target_date,
                "actual": value,
                "hybrid": pending.point,
                "arima": pending.base_point,
                "persistence": pending.persistence,
                "hybrid_error": value - pending.point,
                "arima_error": value - pending.base_point,
                "persistence_error": value - pending.persistence,
                "candidate_fingerprint": self.candidate_fingerprint_,
            }
            self.evaluations_.append(evaluation)
            self.history_ = pd.concat(
                [self.history_, pd.DataFrame({"date": [timestamp], "price": [value]})],
                ignore_index=True,
            )
            self.arima_model_ = self._fit_arima(
                weekly_grid(self.history_)["price"].to_numpy(float)
            )
            self.data_fingerprint_ = self._data_fingerprint(self.history_)
            self.pending_forecast_ = None
            self.pending_scaled_features_ = None
            return dict(evaluation)

    @staticmethod
    def _dm_squared_error(hybrid_error: np.ndarray, base_error: np.ndarray) -> dict[str, float]:
        differential = hybrid_error**2 - base_error**2
        mean = float(np.mean(differential))
        if differential.size < 2:
            return {"statistic": 0.0, "pvalue": 1.0, "mean_loss_difference": mean}
        variance = float(np.var(differential, ddof=1))
        if variance <= 1e-18:
            statistic = 0.0 if abs(mean) <= 1e-12 else np.sign(mean) * np.inf
        else:
            statistic = mean / sqrt(variance / differential.size)
        pvalue = erfc(abs(float(statistic)) / sqrt(2.0))
        return {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "mean_loss_difference": mean,
        }

    def promotion_report(
        self,
        *,
        minimum_outcomes: int = 26,
        preferred_outcomes: int = 52,
    ) -> dict[str, object]:
        """Evaluate frozen gates; this method never promotes automatically."""

        self._require_fitted()
        if minimum_outcomes < 13 or preferred_outcomes < minimum_outcomes:
            raise ValueError("invalid prospective evidence thresholds")
        n = len(self.evaluations_)
        base: dict[str, object] = {
            "n_outcomes": n,
            "minimum_outcomes": minimum_outcomes,
            "preferred_outcomes": preferred_outcomes,
            "automatic_promotion_allowed": False,
            "candidate_fingerprint": self.candidate_fingerprint_,
        }
        if n == 0:
            return {**base, "status": "collecting", "weeks_remaining": minimum_outcomes}
        actual = np.asarray([row["actual"] for row in self.evaluations_], dtype=float)
        hybrid = np.asarray([row["hybrid"] for row in self.evaluations_], dtype=float)
        arima = np.asarray([row["arima"] for row in self.evaluations_], dtype=float)
        persistence = np.asarray(
            [row["persistence"] for row in self.evaluations_], dtype=float
        )
        hybrid_metrics = regression_report(actual, hybrid)
        arima_metrics = regression_report(actual, arima)
        persistence_metrics = regression_report(actual, persistence)
        ratio_arima = hybrid_metrics["rmse"] / max(arima_metrics["rmse"], 1e-12)
        ratio_persistence = hybrid_metrics["rmse"] / max(
            persistence_metrics["rmse"], 1e-12
        )
        worst_13 = None
        if n >= 13:
            ratios = []
            for start in range(n - 12):
                stop = start + 13
                h_rmse = regression_report(actual[start:stop], hybrid[start:stop])["rmse"]
                a_rmse = regression_report(actual[start:stop], arima[start:stop])["rmse"]
                ratios.append(h_rmse / max(a_rmse, 1e-12))
            worst_13 = float(max(ratios))
        dm = self._dm_squared_error(actual - hybrid, actual - arima)
        enough = n >= minimum_outcomes
        gates = {
            "minimum_evidence": enough,
            "rmse_gain_vs_arima_at_least_2pct": enough and ratio_arima <= 0.98,
            "rmse_gain_vs_persistence_at_least_2pct": enough and ratio_persistence <= 0.98,
            "worst_13_week_ratio_below_1_05": enough
            and worst_13 is not None
            and worst_13 <= 1.05,
            "dm_pvalue_below_0_05": enough
            and dm["mean_loss_difference"] < 0
            and dm["pvalue"] < 0.05,
            "preferred_evidence": n >= preferred_outcomes,
        }
        eligible = all(
            gates[name]
            for name in (
                "minimum_evidence",
                "rmse_gain_vs_arima_at_least_2pct",
                "rmse_gain_vs_persistence_at_least_2pct",
                "worst_13_week_ratio_below_1_05",
                "dm_pvalue_below_0_05",
            )
        )
        status = "collecting" if not enough else (
            "eligible_for_human_review" if eligible else "gates_not_met"
        )
        return {
            **base,
            "status": status,
            "weeks_remaining": max(0, minimum_outcomes - n),
            "hybrid_metrics": hybrid_metrics,
            "arima_metrics": arima_metrics,
            "persistence_metrics": persistence_metrics,
            "rmse_ratio_vs_arima": float(ratio_arima),
            "rmse_ratio_vs_persistence": float(ratio_persistence),
            "worst_13_week_rmse_ratio_vs_arima": worst_13,
            "dm_vs_arima": dm,
            "gates": gates,
        }

    def health(self) -> S10ShadowHealth:
        with self._lock:
            self._require_fitted()
            summary = self.vs_model_.summary()
            warning_names: list[str] = []
            if self.last_feature_clip_fraction_ > 0.25:
                warning_names.append("feature_distribution_shift")
            if self.vs_model_.n_rules >= 0.9 * self.vs_model_.config.max_rules:
                warning_names.append("rule_capacity_pressure")
            if int(summary["max_dictionary_size"]) >= 0.9 * self.vs_model_.config.max_dictionary_size:
                warning_names.append("dictionary_capacity_pressure")
            if float(summary["dictionary_replacement_rate"]) >= 0.40:
                warning_names.append("dictionary_replacement_churn")
            return S10ShadowHealth(
                status="warning" if warning_names else "healthy",
                n_observations=len(self.history_),
                n_outcomes=len(self.evaluations_),
                pending_target_date=(
                    self.pending_forecast_.target_date if self.pending_forecast_ else None
                ),
                n_rules=self.vs_model_.n_rules,
                beta=float(self.vs_model_.beta_),
                max_dictionary_size=int(summary["max_dictionary_size"]),
                dictionary_replacements=int(summary["dictionary_replacements"]),
                dictionary_replacement_rate=float(summary["dictionary_replacement_rate"]),
                feature_clip_fraction=self.last_feature_clip_fraction_,
                warnings=tuple(warning_names),
            )

    def metadata(self) -> dict[str, object]:
        self._require_fitted()
        return {
            "artifact_version": self.artifact_version,
            "created_at_utc": self.created_at_utc_,
            "scope": "prospective Diesel B S10 residual hybrid shadow, horizon=1",
            "candidate": asdict(self.candidate),
            "candidate_fingerprint": self.candidate_fingerprint_,
            "freeze_start": self.freeze_start_,
            "freeze_cutoff": self.freeze_cutoff_,
            "data_fingerprint": self.data_fingerprint_,
            "baseline_training_samples": self.baseline_training_samples_,
            "n_outcomes": len(self.evaluations_),
            "pending_forecast_id": (
                self.pending_forecast_.forecast_id if self.pending_forecast_ else None
            ),
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "statsmodels": statsmodels.__version__,
                "joblib": joblib.__version__,
            },
        }

    def save(self, path: str | Path) -> Path:
        with self._lock:
            self._require_fitted()
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            joblib.dump(self, temporary)
            temporary.replace(destination)
            return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        expected_candidate_fingerprint: str | None = None,
    ) -> "S10ResidualHybridShadow":
        source = Path(path)
        if expected_sha256 is not None:
            observed = cls._file_sha256(source)
            if observed.lower() != expected_sha256.lower():
                raise RuntimeError("shadow artifact SHA-256 mismatch")
        model = joblib.load(source)
        if not isinstance(model, cls):
            raise TypeError("artifact is not an S10ResidualHybridShadow")
        if getattr(model, "artifact_version_", None) != cls.artifact_version:
            raise RuntimeError("unsupported shadow artifact version")
        if (
            expected_candidate_fingerprint is not None
            and model.candidate_fingerprint_ != expected_candidate_fingerprint
        ):
            raise RuntimeError("shadow candidate fingerprint mismatch")
        model._require_fitted()
        return model


def _canonical_record(record: dict[str, object]) -> bytes:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def verify_shadow_ledger(path: str | Path) -> list[dict[str, object]]:
    """Read and verify every record in a hash-chained JSONL ledger."""

    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, object]] = []
    previous = "0" * 64
    for expected_sequence, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        observed_hash = str(record.pop("record_hash", ""))
        if record.get("sequence") != expected_sequence:
            raise RuntimeError("shadow ledger sequence mismatch")
        if record.get("previous_hash") != previous:
            raise RuntimeError("shadow ledger chain mismatch")
        expected_hash = hashlib.sha256(_canonical_record(record)).hexdigest()
        if observed_hash != expected_hash:
            raise RuntimeError("shadow ledger record hash mismatch")
        record["record_hash"] = observed_hash
        records.append(record)
        previous = observed_hash
    return records


def append_shadow_ledger(
    path: str | Path,
    *,
    event: Literal["freeze", "forecast", "outcome"],
    payload: dict[str, object],
) -> dict[str, object]:
    """Atomically append one verified event to the shadow ledger."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = verify_shadow_ledger(destination)
    previous = records[-1]["record_hash"] if records else "0" * 64
    record: dict[str, object] = {
        "sequence": len(records) + 1,
        "previous_hash": previous,
        "event": event,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    record["record_hash"] = hashlib.sha256(_canonical_record(record)).hexdigest()
    records.append(record)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            for item in records
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return dict(record)
