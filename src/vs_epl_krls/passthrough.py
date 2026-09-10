"""Modelo de repasse (pass-through) com correcao de erro para o Diesel B S10.

Motivacao empirica
------------------
O preco medio nacional semanal de revenda e quase constante em dois tercos das
semanas e se move em saltos concentrados.  No holdout de 104 semanas usado pelo
repositorio, uma unica semana responde por 75% do erro quadratico do ARIMA.  Um
modelo que so olha a propria serie tem teto estreito: em walk-forward honesto,
momentum puro ganha menos de 2% sobre persistencia, e um oraculo que soubesse
apenas *se* havera evento ganharia +1,5%.  Praticamente todo o valor esta na
*magnitude* do salto, e a magnitude vem de fora da serie.

A fonte causal disponivel e o custo do insumo em reais, ``brent x usdbrl``.  O
indice semanal da ANP e datado pelo domingo que inicia a semana pesquisada, e o
resample do painel usa ``ffill``, entao ``brent_brl`` na linha ``T`` e o
fechamento da sexta anterior ao inicio da semana ``T``: esta estritamente no
passado de toda a janela de medicao do preco.  O repasse observado leva cerca de
duas semanas, de modo que as defasagens 1 e 2 concentram o sinal.

Especificacao
-------------
Alvo ``y(T) = p(T) - p(T-1)`` em R$/L, previsto na origem ``T-1``::

    y(T) = a + b1*y(T-1)
             + b2*dlog_bb(T-1)*p(T-1)
             + b3*dlog_bb(T-2)*p(T-1)
             + g *z_coint(T-1)*p(T-1)

O produto por ``p(T-1)`` converte uma variacao *relativa* do insumo em R$/L, o
que importa numa serie cujo nivel triplicou no periodo.  ``z_coint`` e o residuo
padronizado da regressao de cointegracao ``log p ~ log(brent*usdbrl)``, estimada
em janela expansiva; ele carrega o repasse ainda pendente.

A estimacao usa IRLS de Huber porque minimos quadrados sao dominados por poucas
semanas de choque, o que distorce os coeficientes para as demais.

Todos os ajustes ocorrem em janela expansiva, apenas com informacao anterior a
cada origem.  Nada aqui le o holdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "PassThroughConfig",
    "PassThroughForecast",
    "PassThroughECM",
    "build_passthrough_panel",
    "build_parity_panel",
    "PARITY_FEATURES",
]

_FEATURES: tuple[str, ...] = ("dp1", "rbb1", "rbb2", "coint_r")

#: Atributos do modelo de paridade de diesel.  Todos disponiveis em tempo real
#: na origem da previsao: o ULSD e o dolar fecham antes de a semana comecar.
PARITY_FEATURES: tuple[str, ...] = ("dp1", "rpar1", "rpar2", "coint_par")
_MIN_TRAIN = 60
_COINT_WARMUP = 52


@dataclass(frozen=True)
class PassThroughConfig:
    """Hiperparametros congelados na selecao de desenvolvimento."""

    huber_delta: float = 3.5
    irls_iterations: int = 8
    min_train: int = _MIN_TRAIN
    coint_warmup: int = _COINT_WARMUP
    volatility_window: int = 26
    interval_nominal: float = 0.80
    change_limit_sigma: float = 8.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive")
        if self.irls_iterations < 1:
            raise ValueError("irls_iterations must be at least 1")
        if self.min_train < 30:
            raise ValueError("min_train must be at least 30")
        if not 0.5 <= self.interval_nominal < 1.0:
            raise ValueError("interval_nominal must be in [0.5, 1)")


@dataclass(frozen=True)
class PassThroughForecast:
    """Previsao pontual e intervalo condicional para a proxima semana."""

    origin_date: pd.Timestamp
    target_date: pd.Timestamp
    origin_price: float
    point: float
    delta: float
    lower: float
    upper: float
    nominal_coverage: float
    conditional_sigma: float
    fallback_used: bool
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "origin_date": str(pd.Timestamp(self.origin_date).date()),
            "target_date": str(pd.Timestamp(self.target_date).date()),
            "origin_price": round(float(self.origin_price), 6),
            "point": round(float(self.point), 6),
            "delta": round(float(self.delta), 6),
            "lower": round(float(self.lower), 6),
            "upper": round(float(self.upper), 6),
            "nominal_coverage": float(self.nominal_coverage),
            "conditional_sigma": round(float(self.conditional_sigma), 6),
            "fallback_used": bool(self.fallback_used),
            "reason": self.reason,
        }


def _expanding_cointegration(
    log_price: pd.Series, log_cost: pd.Series, *, warmup: int
) -> pd.Series:
    """Residuo padronizado de ``log p ~ log custo``, estimado so com o passado.

    As somas expansivas usam series ja defasadas em uma semana, de modo que o
    valor na linha ``T`` depende apenas de observacoes ate ``T-1``.
    """

    x = log_cost.shift(1)
    y = log_price.shift(1)
    valid = x.notna() & y.notna()
    xv = x.where(valid)
    yv = y.where(valid)
    n = valid.cumsum()
    sx = xv.expanding().sum()
    sy = yv.expanding().sum()
    sxx = (xv * xv).expanding().sum()
    sxy = (xv * yv).expanding().sum()
    denominator = (n * sxx - sx * sx).replace(0.0, np.nan)
    theta = (n * sxy - sx * sy) / denominator
    alpha = (sy - theta * sx) / n.replace(0, np.nan)
    residual = yv - (alpha + theta * xv)
    scale = residual.expanding(min_periods=warmup).std()
    standardized = residual / scale.replace(0.0, np.nan)
    standardized[n < warmup] = np.nan
    return standardized.replace([np.inf, -np.inf], np.nan)


def build_passthrough_panel(
    frame: pd.DataFrame,
    *,
    config: PassThroughConfig | None = None,
) -> pd.DataFrame:
    """Constroi o painel causal a partir de ``date``, ``price``, ``brent``, ``usdbrl``.

    Toda coluna de atributo na linha ``T`` usa exclusivamente informacao
    disponivel na origem ``T-1``.
    """

    config = config or PassThroughConfig()
    required = {"date", "price", "brent", "usdbrl"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"passthrough panel requires columns: {sorted(missing)}")
    data = frame[["date", "price", "brent", "usdbrl"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("price", "brent", "usdbrl"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = (
        data.dropna(subset=["date", "price"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    if len(data) < config.min_train + config.coint_warmup:
        raise ValueError("passthrough panel needs a longer history")
    if (data["price"] <= 0).any():
        raise ValueError("prices must be strictly positive")

    cost = data["brent"] * data["usdbrl"]
    if (cost <= 0).any():
        raise ValueError("brent and usdbrl must be strictly positive")
    log_price = np.log(data["price"])
    log_cost = np.log(cost)
    previous_price = data["price"].shift(1)

    panel = pd.DataFrame({"date": data["date"], "price": data["price"]})
    panel["origin_price"] = previous_price
    panel["y"] = data["price"].diff()
    panel["dp1"] = panel["y"].shift(1)
    delta_cost = log_cost.diff()
    panel["rbb1"] = delta_cost.shift(1) * previous_price
    panel["rbb2"] = delta_cost.shift(2) * previous_price
    panel["coint_r"] = (
        _expanding_cointegration(log_price, log_cost, warmup=config.coint_warmup)
        * previous_price
    )
    panel["volatility"] = (
        panel["y"].shift(1).rolling(config.volatility_window, min_periods=12).std()
    )
    panel["abs_cost_move"] = (delta_cost.shift(1) * previous_price).abs()
    return panel


def build_parity_panel(
    frame: pd.DataFrame,
    *,
    config: PassThroughConfig | None = None,
    producer_lag: int = 3,
) -> pd.DataFrame:
    """Painel causal baseado na paridade de importacao do diesel (ULSD).

    Espera as colunas produzidas por :func:`vs_epl_krls.causal_ingest.build_causal_panel`:
    ``date``, ``price``, ``parity``, ``brent_brl`` e, opcionalmente,
    ``producer_price``.

    O preco de produtor entra apenas com ``producer_lag`` semanas de defasagem,
    porque a ANP publica o arquivo cerca de doze dias apos o fim da semana de
    competencia.  Nessa defasagem ele quase nao carrega sinal; fica no painel
    para documentacao e diagnostico, nao como atributo principal.
    """

    config = config or PassThroughConfig()
    required = {"date", "price", "parity"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"parity panel requires columns: {sorted(missing)}")
    if producer_lag < 1:
        raise ValueError("producer_lag must be at least one week")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("price", "parity"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = (
        data.dropna(subset=["date", "price", "parity"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    if len(data) < config.min_train + config.coint_warmup:
        raise ValueError("parity panel needs a longer history")
    if (data["price"] <= 0).any() or (data["parity"] <= 0).any():
        raise ValueError("prices and parity must be strictly positive")

    from .weekly import weekly_grid

    data = weekly_grid(data)
    return _parity_features(data, config=config, producer_lag=producer_lag)


def _parity_features(
    data: pd.DataFrame, *, config: PassThroughConfig, producer_lag: int
) -> pd.DataFrame:
    """Shared calendar-aligned features; all predictors use earlier rows."""
    log_price = np.log(data["price"])
    log_parity = np.log(data["parity"])
    previous_price = data["price"].shift(1)

    panel = pd.DataFrame({"date": data["date"], "price": data["price"]})
    panel["origin_price"] = previous_price
    panel["y"] = data["price"].diff()
    panel["dp1"] = panel["y"].shift(1)

    delta_parity = log_parity.diff()
    panel["rpar1"] = delta_parity.shift(1) * previous_price
    panel["rpar2"] = delta_parity.shift(2) * previous_price
    panel["coint_par"] = (
        _expanding_cointegration(log_price, log_parity, warmup=config.coint_warmup)
        * previous_price
    )

    if "brent_brl" in data:
        brent_brl = pd.to_numeric(data["brent_brl"], errors="coerce")
        panel["rbb1"] = np.log(brent_brl).diff().shift(1) * previous_price
    if "producer_price" in data:
        producer = pd.to_numeric(data["producer_price"], errors="coerce")
        panel[f"rprod{producer_lag}"] = (
            np.log(producer).diff().shift(producer_lag) * previous_price
        )

    panel["volatility"] = (
        panel["y"].shift(1).rolling(config.volatility_window, min_periods=12).std()
    )
    panel["abs_cost_move"] = panel["rpar1"].abs()
    return panel


def build_next_parity_row(
    frame: pd.DataFrame, *, config: PassThroughConfig | None = None, producer_lag: int = 3
) -> pd.Series:
    """Build the unobserved next week with exactly the training feature logic.

    No placeholder price or future cost is needed: the new row is all NaN,
    and every predictor depends only on preceding observations.
    """
    from .weekly import weekly_grid

    config = config or PassThroughConfig()
    # Apply the same input validation as training before using the raw history.
    panel = build_parity_panel(frame, config=config, producer_lag=producer_lag)
    data = weekly_grid(frame)
    origin_date = data["date"].iloc[-1]
    if panel["date"].iloc[-1] != origin_date or pd.isna(data["price"].iloc[-1]):
        raise ValueError("latest parity origin must have observed price and parity")
    next_date = origin_date + pd.Timedelta(weeks=1)
    future = pd.DataFrame({"date": [next_date]})
    extended = pd.concat([data, future], ignore_index=True)
    row = _parity_features(extended, config=config, producer_lag=producer_lag).iloc[-1].copy()
    row["origin_date"] = origin_date
    return row


def _huber_irls(
    design: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    delta: float,
    iterations: int,
) -> NDArray[np.float64] | None:
    """Regressao de Huber por minimos quadrados reponderados iterativamente."""

    try:
        beta = np.linalg.solve(design.T @ design, design.T @ target)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    if not np.all(np.isfinite(beta)):
        return None
    for _ in range(iterations):
        residual = target - design @ beta
        median = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - median))
        if not np.isfinite(scale) or scale <= 0:
            break
        weight = np.clip(delta * scale / np.maximum(np.abs(residual), 1e-12), None, 1.0)
        weighted = design * weight[:, None]
        try:
            candidate = np.linalg.solve(design.T @ weighted, design.T @ (target * weight))
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(candidate)):
            break
        if np.max(np.abs(candidate - beta)) < 1e-12:
            beta = candidate
            break
        beta = candidate
    return beta if np.all(np.isfinite(beta)) else None


@dataclass
class PassThroughECM:
    """Modelo de repasse com correcao de erro, ajustado em janela expansiva."""

    config: PassThroughConfig = field(default_factory=PassThroughConfig)
    feature_names: tuple[str, ...] = _FEATURES
    coefficients_: NDArray[np.float64] | None = None
    sigma_coefficients_: NDArray[np.float64] | None = None
    standardized_quantiles_: tuple[float, float] | None = None
    n_train_: int = 0
    panel_: pd.DataFrame | None = None

    # ------------------------------------------------------------------ ajuste
    def fit(self, panel: pd.DataFrame, *, end: int | None = None) -> "PassThroughECM":
        """Ajusta usando apenas as linhas anteriores a ``end`` (exclusivo)."""

        if "y" not in panel.columns:
            raise ValueError("panel must contain the target column 'y'")
        end = len(panel) if end is None else int(end)
        if end <= 0 or end > len(panel):
            raise ValueError("end is outside the panel")
        window = panel.iloc[:end]
        design_raw = window[list(self.feature_names)].to_numpy(float)
        target = window["y"].to_numpy(float)
        usable = np.isfinite(design_raw).all(axis=1) & np.isfinite(target)
        if int(usable.sum()) < self.config.min_train:
            raise ValueError(
                f"not enough usable rows to fit: {int(usable.sum())} < {self.config.min_train}"
            )
        design = np.column_stack([np.ones(int(usable.sum())), design_raw[usable]])
        beta = _huber_irls(
            design,
            target[usable],
            delta=self.config.huber_delta,
            iterations=self.config.irls_iterations,
        )
        if beta is None:
            raise ValueError("robust regression failed to converge to a finite solution")
        self.coefficients_ = beta
        self.n_train_ = int(usable.sum())
        self.panel_ = panel
        self._fit_interval(window, design, target[usable], usable)
        return self

    def _fit_interval(
        self,
        window: pd.DataFrame,
        design: NDArray[np.float64],
        target: NDArray[np.float64],
        usable: NDArray[np.bool_],
    ) -> None:
        """Calibra um desvio condicional e quantis do residuo padronizado.

        Um intervalo de largura fixa e largo demais nas semanas paradas e
        estreito demais nas semanas de choque.  Aqui a escala responde a
        volatilidade recente e ao tamanho do movimento do insumo.
        """

        assert self.coefficients_ is not None
        residual = target - design @ self.coefficients_
        scale_raw = window.loc[usable, ["volatility", "abs_cost_move"]].to_numpy(float)
        finite = np.isfinite(scale_raw).all(axis=1)
        absolute = np.abs(residual)
        if int(finite.sum()) >= self.config.min_train:
            scale_design = np.column_stack([np.ones(int(finite.sum())), scale_raw[finite]])
            gamma, *_ = np.linalg.lstsq(scale_design, absolute[finite], rcond=None)
            if not np.all(np.isfinite(gamma)):
                gamma = None
        else:
            gamma = None
        if gamma is None:
            self.sigma_coefficients_ = None
            floor = float(np.median(absolute)) or 1e-4
            standardized = residual / floor
            self.standardized_quantiles_ = self._quantiles(standardized)
            return
        self.sigma_coefficients_ = gamma
        predicted = scale_design @ gamma
        floor = max(float(np.median(absolute)) * 0.25, 1e-4)
        predicted = np.maximum(predicted, floor)
        self.standardized_quantiles_ = self._quantiles(residual[finite] / predicted)

    def _quantiles(self, standardized: NDArray[np.float64]) -> tuple[float, float]:
        clean = standardized[np.isfinite(standardized)]
        if clean.size < 20:
            return (-1.2816, 1.2816)
        tail = (1.0 - self.config.interval_nominal) / 2.0
        low, high = np.quantile(clean, [tail, 1.0 - tail])
        return (float(low), float(high))

    # ----------------------------------------------------------------- predicao
    def _conditional_sigma(self, row: pd.Series) -> float:
        if self.sigma_coefficients_ is None:
            return float("nan")
        features = np.array(
            [1.0, float(row.get("volatility", np.nan)), float(row.get("abs_cost_move", np.nan))]
        )
        if not np.all(np.isfinite(features)):
            return float("nan")
        return max(float(features @ self.sigma_coefficients_), 1e-4)

    def predict_delta(self, row: pd.Series) -> float:
        """Variacao prevista em R$/L; ``nan`` quando os atributos nao estao prontos."""

        if self.coefficients_ is None:
            raise RuntimeError("model is not fitted")
        values = np.array([float(row.get(name, np.nan)) for name in self.feature_names])
        if not np.all(np.isfinite(values)):
            return float("nan")
        delta = float(np.concatenate([[1.0], values]) @ self.coefficients_)
        return delta if np.isfinite(delta) else float("nan")

    def forecast_row(self, row: pd.Series, *, origin_price: float) -> PassThroughForecast:
        """Previsao para uma linha do painel, com fallback para persistencia."""

        if not np.isfinite(origin_price) or origin_price <= 0:
            raise ValueError("origin_price must be finite and positive")
        delta = self.predict_delta(row)
        sigma = self._conditional_sigma(row)
        fallback, reason = False, ""
        if not np.isfinite(delta):
            delta, fallback, reason = 0.0, True, "features_unavailable"
        else:
            limit = self._change_limit(row)
            if np.isfinite(limit) and abs(delta) > limit:
                delta, fallback, reason = 0.0, True, "implausible_change"
        if not np.isfinite(sigma):
            sigma = float(row.get("volatility", np.nan))
            if not np.isfinite(sigma):
                sigma = 0.02
        low, high = self.standardized_quantiles_ or (-1.2816, 1.2816)
        point = float(origin_price + delta)
        return PassThroughForecast(
            origin_date=row.get("origin_date", pd.NaT),
            target_date=row.get("date", pd.NaT),
            origin_price=float(origin_price),
            point=point,
            delta=float(delta),
            lower=float(point + low * sigma),
            upper=float(point + high * sigma),
            nominal_coverage=self.config.interval_nominal,
            conditional_sigma=float(sigma),
            fallback_used=fallback,
            reason=reason,
        )

    def _change_limit(self, row: pd.Series) -> float:
        volatility = float(row.get("volatility", np.nan))
        if not np.isfinite(volatility) or volatility <= 0:
            return float("nan")
        return self.config.change_limit_sigma * volatility

    # -------------------------------------------------------------- walkforward
    def walk_forward(
        self,
        panel: pd.DataFrame,
        start: int,
        end: int,
        *,
        refit_every: int = 1,
    ) -> pd.DataFrame:
        """Reajusta em janela expansiva e preve ``[start, end)`` sem olhar o futuro."""

        if not 0 < start < end <= len(panel):
            raise ValueError("invalid walk-forward window")
        records: list[dict[str, object]] = []
        model = PassThroughECM(config=self.config, feature_names=self.feature_names)
        last_fit = -(10**9)
        for index in range(start, end):
            if index - last_fit >= refit_every:
                try:
                    model.fit(panel, end=index)
                    last_fit = index
                except ValueError:
                    pass
            row = panel.iloc[index]
            origin_price = float(row["origin_price"])
            if model.coefficients_ is None or not np.isfinite(origin_price):
                records.append(
                    {
                        "date": row["date"],
                        "actual": float(row["price"]),
                        "prediction": np.nan,
                        "persistence": origin_price,
                        "lower": np.nan,
                        "upper": np.nan,
                        "sigma": np.nan,
                        "fallback": True,
                    }
                )
                continue
            forecast = model.forecast_row(row, origin_price=origin_price)
            records.append(
                {
                    "date": row["date"],
                    "actual": float(row["price"]),
                    "prediction": forecast.point,
                    "persistence": origin_price,
                    "lower": forecast.lower,
                    "upper": forecast.upper,
                    "sigma": forecast.conditional_sigma,
                    "fallback": forecast.fallback_used,
                }
            )
        return pd.DataFrame.from_records(records)

    # ------------------------------------------------------------------ estado
    def summary(self) -> dict[str, object]:
        if self.coefficients_ is None:
            return {"fitted": False}
        names = ("intercept",) + tuple(self.feature_names)
        return {
            "fitted": True,
            "n_train": self.n_train_,
            "coefficients": {
                name: round(float(value), 8)
                for name, value in zip(names, self.coefficients_)
            },
            "standardized_quantiles": [
                round(float(value), 6) for value in (self.standardized_quantiles_ or ())
            ],
            "conditional_sigma_model": (
                None
                if self.sigma_coefficients_ is None
                else {
                    name: round(float(value), 8)
                    for name, value in zip(
                        ("intercept", "volatility", "abs_cost_move"),
                        self.sigma_coefficients_,
                    )
                }
            ),
            "config": {
                "huber_delta": self.config.huber_delta,
                "interval_nominal": self.config.interval_nominal,
                "coint_warmup": self.config.coint_warmup,
                "volatility_window": self.config.volatility_window,
            },
        }
