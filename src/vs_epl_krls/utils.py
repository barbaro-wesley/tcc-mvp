"""Validation and online scaling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_finite_vector(
    x: ArrayLike,
    *,
    name: str = "x",
    expected_features: int | None = None,
) -> NDArray[np.float64]:
    """Convert one sample to a validated 1-D float vector."""

    value = np.asarray(x, dtype=float)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if expected_features is not None and value.size != expected_features:
        raise ValueError(
            f"{name} has {value.size} features; expected {expected_features}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return value.astype(float, copy=False)


def as_finite_target(y: float) -> float:
    """Validate a scalar regression target."""

    value = np.asarray(y, dtype=float)
    if value.ndim != 0 or not np.isfinite(value):
        raise ValueError("y must be a finite scalar")
    return float(value)


@dataclass
class RunningTargetStats:
    """Leak-free running target statistics used for error normalization."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = np.inf
    maximum: float = -np.inf

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def range(self) -> float:
        return 0.0 if self.count < 2 else float(self.maximum - self.minimum)

    @property
    def std(self) -> float:
        return 0.0 if self.count < 2 else float(np.sqrt(self.m2 / (self.count - 1)))


class RobustBoundedScaler:
    """Past-fitted median/IQR scaling into (0,1), without hard clipping.

    The arctangent retains ordering beyond the historical range. The scaler
    remains frozen with the fuzzy centers and KRLS dictionaries it encodes.
    """

    def fit(self, x: ArrayLike) -> "RobustBoundedScaler":
        values = np.asarray(x, dtype=float)
        if values.ndim != 2 or not values.size or not np.isfinite(values).all():
            raise ValueError("x must be a non-empty finite 2-D array")
        self.center_ = np.median(values, axis=0)
        q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
        spread = q75 - q25
        fallback = np.std(values, axis=0)
        self.scale_ = np.where(spread > 1e-12, spread, np.where(fallback > 1e-12, fallback, 1.0))
        return self

    def transform(self, x: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "center_"):
            raise RuntimeError("the scaler has not been fitted")
        values = np.asarray(x, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.center_.size:
            raise ValueError("x has an incompatible shape")
        if not np.isfinite(values).all():
            raise ValueError("x must be finite")
        with np.errstate(over="ignore"):
            return 0.5 + np.arctan((values - self.center_) / self.scale_) / np.pi


class MinMaxScaler:
    """Small NumPy-only min-max scaler fitted on an explicit training split."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        lower, upper = feature_range
        if not np.isfinite(lower + upper) or upper <= lower:
            raise ValueError("feature_range must be finite and increasing")
        self.feature_range = (float(lower), float(upper))
        self.data_min_: NDArray[np.float64] | None = None
        self.data_max_: NDArray[np.float64] | None = None

    def fit(self, x: ArrayLike) -> "MinMaxScaler":
        values = np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("x must be a non-empty finite 1-D or 2-D array")
        self.data_min_ = np.min(values, axis=0)
        self.data_max_ = np.max(values, axis=0)
        return self

    def transform(self, x: ArrayLike) -> NDArray[np.float64]:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("the scaler has not been fitted")
        values = np.asarray(x, dtype=float)
        original_1d = values.ndim == 1
        if original_1d:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[1] != self.data_min_.size:
            raise ValueError("x has an incompatible shape")
        scale = np.where(self.data_max_ > self.data_min_, self.data_max_ - self.data_min_, 1.0)
        lower, upper = self.feature_range
        result = lower + (values - self.data_min_) * (upper - lower) / scale
        return result.ravel() if original_1d else result

    def inverse_transform(self, x: ArrayLike) -> NDArray[np.float64]:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("the scaler has not been fitted")
        values = np.asarray(x, dtype=float)
        original_1d = values.ndim == 1
        if original_1d:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[1] != self.data_min_.size:
            raise ValueError("x has an incompatible shape")
        lower, upper = self.feature_range
        data_scale = self.data_max_ - self.data_min_
        result = self.data_min_ + (values - lower) * data_scale / (upper - lower)
        return result.ravel() if original_1d else result
