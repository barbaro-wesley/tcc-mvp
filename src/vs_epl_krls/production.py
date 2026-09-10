"""Production bundle for national weekly Diesel B S10 forecasting.

The bundle packages preprocessing, the selected ARIMA primary, the calibrated
VS-ePL-KRLS challenger, Ridge, persistence, empirical residual intervals, guardrails,
health metadata and atomic serialization.  Scope is intentionally restricted to
one-week-ahead S10 forecasts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import platform
from pathlib import Path
import threading
from typing import Literal
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import statsmodels
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from .model import VSEPLKRLS
from .selection import (
    S10Candidate,
    build_s10_feature_frame,
    build_s10_supervised,
)
from .utils import MinMaxScaler
from .weekly import weekly_grid


PrimaryModel = Literal["ARIMA", "ensemble", "Ridge", "VS-ePL-KRLS", "persistencia"]


@dataclass(frozen=True)
class S10Forecast:
    target_date: str
    point: float
    p10: float
    p90: float
    primary_model: str
    fallback_used: bool
    fallback_reason: str | None
    components: dict[str, float]
    last_observed_date: str
    last_observed_price: float
    data_fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class S10HealthReport:
    status: Literal["healthy", "warning"]
    n_observations: int
    last_date: str
    n_rules: int
    beta: float
    max_dictionary_size: int
    feature_clip_fraction: float
    rule_capacity_fraction: float
    dictionary_capacity_fraction: float
    dictionary_replacements: int
    dictionary_replacement_rate: float
    interval_monitor_samples: int
    empirical_interval_coverage: float | None
    recent_mae: float | None
    last_cadence_days: int
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class S10ProductionForecaster:
    """Serializable, thread-safe S10 one-week forecasting bundle."""

    artifact_version = "1.2.0"

    #: Contratos que o carregador aceita.  A 1.1.0 continua servivel: ela nao tem
    #: estado conformal, e sem ele o intervalo cai exatamente no quantil fixo de
    #: antes — previsao identica, byte a byte.
    supported_artifact_versions = ("1.1.0", "1.2.0")

    #: Nivel nominal do intervalo servido.
    interval_nominal_coverage = 0.80

    #: Passo do ajuste conformal adaptativo (Gibbs e Candes, 2021).  O intervalo
    #: do bundle era montado com quantis fixos 0,10/0,90 e entregou **92,3% de
    #: cobertura para um nominal de 80%** no holdout.  Banda larga demais nao e
    #: seguranca gratuita: ela desloca o cenario P90 e distorce o custo aparente
    #: de antecipar.  Aqui o nivel de miscobertura passa a ser aprendido semana a
    #: semana, com os quantis do residuo continuando a vir da janela movel.
    interval_gamma = 0.02

    #: Limites de ``alpha``.  O artigo permite intervalo infinito quando alpha
    #: chega a zero; num produto isso e inutil, entao o valor e limitado e cada
    #: passo em que o limite atuou fica contado.
    interval_alpha_bounds = (0.02, 0.50)
    component_order = ("ARIMA", "Ridge", "persistencia", "VS-ePL-KRLS")

    def __init__(
        self,
        candidate: S10Candidate,
        *,
        primary_model: PrimaryModel = "ARIMA",
        ensemble_weights: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        calibration_residuals: np.ndarray | list[float] | None = None,
        fallback_calibration_residuals: np.ndarray | list[float] | None = None,
        calibration_window: int = 156,
        expected_frequency_days: int = 7,
    ) -> None:
        if candidate.feature_set not in {"price", "lags", "dynamics"}:
            raise ValueError("unsupported candidate feature set")
        if primary_model not in {"ARIMA", "ensemble", "Ridge", "VS-ePL-KRLS", "persistencia"}:
            raise ValueError("unsupported primary_model")
        weights = np.asarray(ensemble_weights, dtype=float)
        if weights.shape != (4,) or np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("ensemble_weights must contain four finite non-negative values")
        if not np.isclose(weights.sum(), 1.0, atol=1e-9):
            raise ValueError("ensemble_weights must sum to one")
        def validated_residuals(
            values: np.ndarray | list[float] | None,
            name: str,
        ) -> np.ndarray:
            residuals = np.asarray([] if values is None else values, dtype=float).ravel()
            if residuals.size and (
                residuals.size < 20 or not np.all(np.isfinite(residuals))
            ):
                raise ValueError(f"{name} must contain at least 20 finite values")
            return residuals

        residuals = validated_residuals(calibration_residuals, "calibration_residuals")
        fallback_residuals = validated_residuals(
            fallback_calibration_residuals,
            "fallback_calibration_residuals",
        )
        if expected_frequency_days < 1:
            raise ValueError("expected_frequency_days must be positive")
        if calibration_window < 20:
            raise ValueError("calibration_window must be at least 20")
        self.candidate = candidate
        self.primary_model = primary_model
        self.ensemble_weights = weights
        self.calibration_window = int(calibration_window)
        self.calibration_residuals = residuals[-self.calibration_window :]
        self.fallback_calibration_residuals = fallback_residuals[
            -self.calibration_window :
        ]
        self.expected_frequency_days = int(expected_frequency_days)
        self.artifact_version_ = self.artifact_version
        self.created_at_utc_ = datetime.now(timezone.utc).isoformat()
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
    def _fingerprint(history: pd.DataFrame) -> str:
        hashes = pd.util.hash_pandas_object(history[["date", "price"]], index=False).to_numpy()
        return hashlib.sha256(hashes.tobytes()).hexdigest()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _fit_arima(prices: np.ndarray):
        best = None
        best_aic = np.inf
        for order in ((0, 1, 1), (1, 1, 0), (1, 1, 1)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    result = ARIMA(prices, order=order, trend="n").fit(
                        method_kwargs={"warn_convergence": False}
                    )
                if np.isfinite(result.aic) and result.aic < best_aic:
                    best = result
                    best_aic = float(result.aic)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
        if best is None:
            raise RuntimeError("all ARIMA candidates failed")
        return best

    def _scaled_features(self, raw: np.ndarray) -> tuple[np.ndarray, float]:
        transformed = self.x_scaler_.transform(raw)
        clipped = np.clip(transformed, 0.0, 1.0)
        fraction = float(np.mean(np.abs(transformed - clipped) > 1e-12))
        return clipped, fraction

    def fit(self, history: pd.DataFrame) -> "S10ProductionForecaster":
        """Fit every component after model selection has been frozen."""

        with self._lock:
            self.history_ = self._validate_history(history)
            supervised = build_s10_supervised(
                self.history_,
                horizon=1,
                feature_set=self.candidate.feature_set,
            )
            self.feature_names_ = supervised.feature_names
            self.x_scaler_ = self.candidate.make_feature_scaler().fit(supervised.x)
            x_scaled, _ = self._scaled_features(supervised.x)
            self.target_scaler_: MinMaxScaler | None = None
            if self.candidate.target_mode == "level":
                self.target_scaler_ = MinMaxScaler().fit(supervised.target_price)
                model_target = self.target_scaler_.transform(supervised.target_price)
            else:
                model_target = supervised.target_price - supervised.origin_price
            self.vs_model_ = VSEPLKRLS(**self.candidate.model_parameters())
            self.vs_model_.fit(x_scaled, model_target)
            self.ridge_model_ = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            self.ridge_model_.fit(supervised.x, supervised.target_price)
            prices = weekly_grid(self.history_)["price"].to_numpy(float)
            self.arima_model_ = self._fit_arima(prices)
            self.data_fingerprint_ = self._fingerprint(self.history_)
            self.training_start_ = str(self.history_["date"].iloc[0].date())
            self.training_end_ = str(self.history_["date"].iloc[-1].date())
            self.last_feature_clip_fraction_ = 0.0
            self.last_cadence_days_ = self.expected_frequency_days
            self.n_online_updates_ = 0
            self.last_forecast_residual_: float | None = None
            self.online_interval_hits_: list[bool] = []
            self.online_absolute_errors_: list[float] = []
            self.is_fitted_ = True
            return self

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("the production forecaster has not been fitted")

    def _latest_raw_features(self) -> np.ndarray:
        frame = build_s10_feature_frame(
            self.history_,
            feature_set=self.candidate.feature_set,
        )
        if frame.empty or frame["date"].iloc[-1] != self.history_["date"].iloc[-1]:
            raise ValueError("latest week needs 13 consecutive observed weeks of features")
        return frame[list(self.feature_names_)].iloc[-1].to_numpy(float).reshape(1, -1)

    def _component_predictions(self) -> tuple[dict[str, float], float]:
        raw = self._latest_raw_features()
        scaled, clip_fraction = self._scaled_features(raw)
        self.last_feature_clip_fraction_ = clip_fraction
        last_price = float(self.history_["price"].iloc[-1])
        vs_output = float(self.vs_model_.predict_one(scaled[0]))
        if self.target_scaler_ is not None:
            vs_price = float(self.target_scaler_.inverse_transform([vs_output])[0])
        else:
            vs_price = last_price + vs_output
        components = {
            "ARIMA": float(np.asarray(self.arima_model_.forecast(steps=1)).ravel()[0]),
            "Ridge": float(self.ridge_model_.predict(raw)[0]),
            "persistencia": last_price,
            "VS-ePL-KRLS": vs_price,
        }
        components["ensemble"] = float(
            sum(
                self.ensemble_weights[index] * components[name]
                for index, name in enumerate(self.component_order)
            )
        )
        return components, clip_fraction

    def _change_limit(self) -> float:
        changes = np.abs(np.diff(self.history_["price"].to_numpy(float)))
        recent = changes[-156:] if changes.size > 156 else changes
        return float(max(0.10, 2.0 * np.quantile(recent, 0.995)))

    def warm_start_interval_alpha(self) -> float:
        """Aquece o nivel do intervalo com os residuos que o bundle ja carrega.

        Sem isto, uma release nova comecaria no nivel nominal e levaria dezenas de
        semanas para descobrir sozinha o que a evidencia ja diz: que o quantil
        fixo cobre demais.  O aquecimento roda o mesmo passo conformal sobre a
        janela de calibracao, em ordem, cada semana usando apenas os residuos
        anteriores a ela.  Nada de futuro entra.
        """

        residuals = np.asarray(self.calibration_residuals, dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size < 40:
            return self._interval_alpha()

        target_alpha = 1.0 - self.interval_nominal_coverage
        low_bound, high_bound = self.interval_alpha_bounds
        alpha = target_alpha
        warmup = 20
        for index in range(warmup, residuals.size):
            past = residuals[:index]
            tail = alpha / 2.0
            low, high = np.quantile(past, [tail, 1.0 - tail])
            covered = bool(low <= residuals[index] <= high)
            alpha = float(
                np.clip(
                    alpha + self.interval_gamma * (target_alpha - (0.0 if covered else 1.0)),
                    low_bound,
                    high_bound,
                )
            )
        self.interval_alpha_ = alpha
        self.interval_alpha_warm_started_ = True
        return alpha

    def _interval_alpha(self) -> float:
        """Nivel de miscobertura corrente, com o padrao dos artefatos antigos."""

        alpha = getattr(self, "interval_alpha_", None)
        if alpha is None or not np.isfinite(alpha):
            return 1.0 - self.interval_nominal_coverage
        return float(alpha)

    def predict_next(self) -> S10Forecast:
        """Forecast the next weekly S10 observation without changing state."""

        with self._lock:
            self._require_fitted()
            components, _ = self._component_predictions()
            point = float(components[self.primary_model])
            last_price = float(self.history_["price"].iloc[-1])
            fallback = False
            reason: str | None = None
            if not np.isfinite(point):
                fallback, reason, point = True, "non_finite_primary", components["persistencia"]
            elif abs(point - last_price) > self._change_limit():
                fallback, reason = True, "implausible_weekly_change"
                arima = components["ARIMA"]
                arima_is_safe = np.isfinite(arima) and abs(arima - last_price) <= self._change_limit()
                point = arima if self.primary_model != "ARIMA" and arima_is_safe else components["persistencia"]
            interval_residuals = (
                self.fallback_calibration_residuals
                if fallback and self.fallback_calibration_residuals.size
                else self.calibration_residuals
            )
            if interval_residuals.size >= 20:
                tail = self._interval_alpha() / 2.0
                low, high = np.quantile(interval_residuals, [tail, 1.0 - tail])
                p10 = max(0.0, min(point, float(point + low)))
                p90 = max(point, float(point + high))
            else:
                p10 = point
                p90 = point
            last_date = pd.Timestamp(self.history_["date"].iloc[-1])
            target_date = last_date + pd.Timedelta(days=self.expected_frequency_days)
            return S10Forecast(
                target_date=str(target_date.date()),
                point=float(point),
                p10=float(p10),
                p90=float(p90),
                primary_model=self.primary_model,
                fallback_used=fallback,
                fallback_reason=reason,
                components=components,
                last_observed_date=str(last_date.date()),
                last_observed_price=last_price,
                data_fingerprint=self.data_fingerprint_,
            )

    def update_one(
        self,
        date: str | pd.Timestamp,
        price: float,
        *,
        allow_anomalous_change: bool = False,
    ) -> "S10ProductionForecaster":
        """Consume the newly observed week and update the fuzzy challenger online."""

        with self._lock:
            self._require_fitted()
            timestamp = pd.Timestamp(date)
            value = float(price)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("price must be finite and positive")
            last_date = pd.Timestamp(self.history_["date"].iloc[-1])
            if timestamp <= last_date:
                raise ValueError("new observation date must be after the current history")
            cadence_days = int((timestamp - last_date).days)
            if timestamp - last_date != pd.Timedelta(days=self.expected_frequency_days):
                raise ValueError("new observation must match the forecast target date; missing week")
            prior_price = float(self.history_["price"].iloc[-1])
            if not allow_anomalous_change and abs(value - prior_price) > self._change_limit():
                raise ValueError(
                    "observed price change exceeds the robust limit; verify source/unit "
                    "or pass allow_anomalous_change=True after human approval"
                )
            issued_forecast = self.predict_next()
            raw = self._latest_raw_features()
            scaled, clip_fraction = self._scaled_features(raw)
            target = value
            if self.target_scaler_ is not None:
                target = float(self.target_scaler_.transform([value])[0])
            else:
                target = value - prior_price
            self.vs_model_.learn_one(scaled[0], target)
            self.history_ = pd.concat(
                [
                    self.history_,
                    pd.DataFrame({"date": [timestamp], "price": [value]}),
                ],
                ignore_index=True,
            )
            supervised = build_s10_supervised(
                self.history_,
                horizon=1,
                feature_set=self.candidate.feature_set,
            )
            self.ridge_model_.fit(supervised.x, supervised.target_price)
            self.arima_model_ = self._fit_arima(weekly_grid(self.history_)["price"].to_numpy(float))
            self.data_fingerprint_ = self._fingerprint(self.history_)
            self.training_end_ = str(timestamp.date())
            self.last_feature_clip_fraction_ = clip_fraction
            self.last_cadence_days_ = cadence_days
            self.n_online_updates_ += 1
            self.last_forecast_residual_ = float(value - issued_forecast.point)
            covered = bool(issued_forecast.p10 <= value <= issued_forecast.p90)
            self.online_interval_hits_.append(covered)
            # Passo conformal adaptativo: cobriu, alpha sobe e a banda encolhe;
            # furou, alpha cai e a banda abre.  A garantia e de longo prazo e nao
            # supoe nada sobre a distribuicao — que e a hipotese que uma serie de
            # combustivel viola o tempo todo.
            target_alpha = 1.0 - self.interval_nominal_coverage
            proposed = self._interval_alpha() + self.interval_gamma * (
                target_alpha - (0.0 if covered else 1.0)
            )
            low_bound, high_bound = self.interval_alpha_bounds
            clipped = float(np.clip(proposed, low_bound, high_bound))
            if clipped != proposed:
                self.interval_alpha_clips_ = getattr(self, "interval_alpha_clips_", 0) + 1
            self.interval_alpha_ = clipped
            self.online_absolute_errors_.append(abs(self.last_forecast_residual_))
            self.online_interval_hits_ = self.online_interval_hits_[
                -self.calibration_window :
            ]
            self.online_absolute_errors_ = self.online_absolute_errors_[
                -self.calibration_window :
            ]
            if issued_forecast.fallback_used:
                self.fallback_calibration_residuals = np.append(
                    self.fallback_calibration_residuals,
                    self.last_forecast_residual_,
                )[-self.calibration_window :]
            else:
                self.calibration_residuals = np.append(
                    self.calibration_residuals,
                    self.last_forecast_residual_,
                )[-self.calibration_window :]
            return self

    def health(self) -> S10HealthReport:
        with self._lock:
            self._require_fitted()
            summary = self.vs_model_.summary()
            rule_fraction = self.vs_model_.n_rules / self.vs_model_.config.max_rules
            dictionary_fraction = (
                float(summary["max_dictionary_size"])
                / self.vs_model_.config.max_dictionary_size
            )
            warnings: list[str] = []
            if self.last_feature_clip_fraction_ > 0.25:
                warnings.append("feature_distribution_shift")
            if rule_fraction >= 0.90:
                warnings.append("rule_capacity_pressure")
            if dictionary_fraction >= 0.90:
                warnings.append("dictionary_capacity_pressure")
            replacement_rate = float(summary["dictionary_replacement_rate"])
            if replacement_rate >= 0.10:
                warnings.append("dictionary_replacement_churn")
            if self.vs_model_.beta_ <= self.vs_model_.config.beta_min * 1.10:
                warnings.append("beta_floor_pressure")
            interval_samples = len(self.online_interval_hits_)
            interval_coverage = (
                float(np.mean(self.online_interval_hits_))
                if interval_samples
                else None
            )
            recent_mae = (
                float(np.mean(self.online_absolute_errors_))
                if self.online_absolute_errors_
                else None
            )
            if (
                interval_samples >= 20
                and interval_coverage is not None
                and interval_coverage < 0.70
            ):
                warnings.append("interval_coverage_degradation")
            if abs(self.last_cadence_days_ - self.expected_frequency_days) > 2:
                warnings.append("unexpected_observation_cadence")
            return S10HealthReport(
                status="warning" if warnings else "healthy",
                n_observations=len(self.history_),
                last_date=str(pd.Timestamp(self.history_["date"].iloc[-1]).date()),
                n_rules=self.vs_model_.n_rules,
                beta=float(self.vs_model_.beta_),
                max_dictionary_size=int(summary["max_dictionary_size"]),
                feature_clip_fraction=self.last_feature_clip_fraction_,
                rule_capacity_fraction=float(rule_fraction),
                dictionary_capacity_fraction=float(dictionary_fraction),
                dictionary_replacements=int(summary["dictionary_replacements"]),
                dictionary_replacement_rate=replacement_rate,
                interval_monitor_samples=interval_samples,
                empirical_interval_coverage=interval_coverage,
                recent_mae=recent_mae,
                last_cadence_days=self.last_cadence_days_,
                warnings=tuple(warnings),
            )

    def metadata(self) -> dict[str, object]:
        self._require_fitted()
        return {
            "artifact_version": self.artifact_version,
            "created_at_utc": self.created_at_utc_,
            "scope": "Diesel B S10, weekly national ANP resale price, horizon=1",
            "primary_model": self.primary_model,
            "challenger": "VS-ePL-KRLS",
            "candidate": asdict(self.candidate),
            "ensemble_weights": self.ensemble_weights.tolist(),
            "training_start": self.training_start_,
            "training_end": self.training_end_,
            "n_observations": len(self.history_),
            "data_fingerprint": self.data_fingerprint_,
            "feature_names": list(self.feature_names_),
            "online_updates": self.n_online_updates_,
            "calibration_samples": int(self.calibration_residuals.size),
            "fallback_calibration_samples": int(
                self.fallback_calibration_residuals.size
            ),
            "calibration_window": self.calibration_window,
            "last_forecast_residual": self.last_forecast_residual_,
            "interval_alpha": round(self._interval_alpha(), 6),
            "interval_alpha_warm_started": bool(
                getattr(self, "interval_alpha_warm_started_", False)
            ),
            "interval_monitor_samples": len(self.online_interval_hits_),
            "empirical_interval_coverage": (
                float(np.mean(self.online_interval_hits_))
                if self.online_interval_hits_
                else None
            ),
            "recent_mae": (
                float(np.mean(self.online_absolute_errors_))
                if self.online_absolute_errors_
                else None
            ),
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "statsmodels": statsmodels.__version__,
                "joblib": joblib.__version__,
            },
        }

    def save(self, path: str | Path) -> Path:
        """Atomically persist the complete fitted bundle."""

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
    ) -> "S10ProductionForecaster":
        """Load a trusted artifact, optionally enforcing its SHA-256 first.

        Joblib uses pickle internally.  Never load an artifact from an
        untrusted source, even when a hash was not supplied.
        """

        source = Path(path)
        if expected_sha256 is not None:
            observed = cls._file_sha256(source)
            if observed.lower() != expected_sha256.lower():
                raise RuntimeError("artifact SHA-256 mismatch")
        model = joblib.load(source)
        if not isinstance(model, cls):
            raise TypeError("artifact is not an S10ProductionForecaster")
        if getattr(model, "artifact_version_", None) not in cls.supported_artifact_versions:
            raise RuntimeError("unsupported artifact version")
        model._require_fitted()
        return model
