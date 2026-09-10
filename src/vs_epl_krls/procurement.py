"""Causal procurement-policy replay for weekly S10 forecasts.

This module measures a narrow, explicit decision policy: when the forecast for
``h`` weeks ahead exceeds today's price by a predeclared threshold, a
configurable share of one week's demand is bought ``h`` weeks early.  It never
changes historical predictions and never uses the future price to decide
whether to buy.

Horizonte
---------
``simulate_horizon_prebuy`` replica a politica para qualquer ``horizon >= 1``.
``simulate_one_week_prebuy`` e o caso ``horizon=1`` e continua sendo a entrada
usada pelos backtests de h=1 ja publicados; os numeros que ela devolve nao
mudaram.

Duas quantidades dependem do horizonte e por isso sao explicitas aqui:

* **carrego** -- antecipar ``h`` semanas paga ``h`` semanas de estoque, entao o
  custo de carregamento entra multiplicado por ``h``.  Ignorar isso faria um
  horizonte maior parecer gratuitamente melhor.
* **anualizacao** -- as decisoes de horizonte ``h`` se sobrepoem: a origem
  ``i`` e a origem ``i+1`` cobrem semanas de consumo que se cruzam.  A media por
  decisao continua sendo multiplicada por 52 porque cada linha ainda representa
  uma semana de consumo, mas o IC90 usa blocos de pelo menos ``h`` semanas para
  nao tratar decisoes sobrepostas como independentes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProcurementBacktest:
    model: str
    period_start: str
    period_end: str
    n_weeks: int
    #: Semanas de antecedencia da decisao.  1 reproduz a politica original.
    horizon_weeks: int
    monthly_liters: float
    weekly_liters: float
    flexibility_fraction: float
    signal_threshold_brl_per_liter: float
    carrying_cost_brl_per_liter_week: float
    triggered_prebuys: int
    trigger_precision: float | None
    liters_shifted: float
    baseline_cost_brl: float
    policy_cost_brl: float
    net_savings_brl: float
    savings_excluding_largest_event_brl: float
    largest_event_share_of_savings: float | None
    savings_per_liter_consumed_brl: float
    annualized_savings_brl: float
    annualized_savings_ci90_brl: tuple[float, float]
    #: Economia liquida semana a semana, zero quando a politica nao disparou.
    #: E a serie que os gates decidiveis reamostram em blocos; sem ela o unico
    #: numero disponivel seria o total, que nao permite medir incerteza.
    weekly_net_savings_brl: tuple[float, ...]
    events: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _annualized_block_bootstrap_ci(
    savings: np.ndarray,
    *,
    block_size: int = 4,
    n_resamples: int = 2000,
    random_state: int = 42,
    horizon: int = 1,
) -> tuple[float, float]:
    # Decisoes de horizonte ``h`` se sobrepoem em ``h-1`` semanas de consumo.  Um
    # bloco menor que ``h`` quebraria essa dependencia e estreitaria o IC de
    # forma artificial, entao o bloco nunca desce abaixo do horizonte.
    block_size = max(int(block_size), int(horizon))
    if savings.size < 8 or n_resamples < 100:
        value = float(np.mean(savings) * 52.0) if savings.size else 0.0
        return value, value
    block = min(max(1, int(block_size)), savings.size)
    starts = np.arange(savings.size - block + 1)
    rng = np.random.default_rng(random_state)
    estimates = np.empty(n_resamples, dtype=float)
    blocks_needed = int(np.ceil(savings.size / block))
    for sample in range(n_resamples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        replay = np.concatenate([savings[start : start + block] for start in chosen])[
            : savings.size
        ]
        estimates[sample] = float(np.mean(replay) * 52.0)
    low, high = np.quantile(estimates, [0.05, 0.95])
    return float(low), float(high)


def simulate_horizon_prebuy(
    predictions: pd.DataFrame,
    *,
    horizon: int = 1,
    prediction_column: str = "arima",
    model_name: str = "ARIMA",
    monthly_liters: float = 200_000.0,
    flexibility_fraction: float = 0.25,
    signal_threshold_brl_per_liter: float = 0.01,
    carrying_cost_brl_per_liter_week: float = 0.0,
    random_state: int = 42,
) -> ProcurementBacktest:
    """Replay a prespecified ``horizon``-week prebuy policy, causally.

    ``persistence`` is the price known at each forecast origin.  Row ``i+h``
    therefore contains the forecast that was available at realized week ``i``
    for realized week ``i+h``.  This alignment is checked explicitly against the
    price realized ``h`` weeks earlier, so a mislabelled file fails loudly
    instead of quietly scoring a forecast that peeked.

    Antecipar ``h`` semanas carrega o volume por ``h`` semanas: o custo de
    carregamento entra multiplicado pelo horizonte.
    """

    if horizon < 1:
        raise ValueError("horizon must be positive")
    horizon = int(horizon)

    required = {"target_date", "actual", "persistence", prediction_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing required columns: {sorted(missing)}")
    if not np.isfinite(monthly_liters) or monthly_liters <= 0:
        raise ValueError("monthly_liters must be finite and positive")
    if not 0.0 <= flexibility_fraction <= 1.0:
        raise ValueError("flexibility_fraction must be between zero and one")
    if signal_threshold_brl_per_liter < 0 or carrying_cost_brl_per_liter_week < 0:
        raise ValueError("threshold and carrying cost cannot be negative")

    frame = predictions[list(required)].copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    for column in ("actual", "persistence", prediction_column):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if len(frame) < 3 or frame.isna().any().any():
        raise ValueError("predictions must contain at least three complete rows")
    if frame["target_date"].duplicated().any() or not frame["target_date"].is_monotonic_increasing:
        raise ValueError("target dates must be unique and chronological")
    values = frame[["actual", "persistence", prediction_column]].to_numpy(float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("prices and predictions must be finite and positive")
    gaps = frame["target_date"].diff().iloc[1:]
    if not (gaps == pd.Timedelta(days=7)).all():
        raise ValueError("procurement replay requires uninterrupted weekly forecasts")
    if len(frame) <= horizon:
        raise ValueError("predictions must be longer than the horizon")
    # A origem da linha ``i+h`` e o preco realizado na linha ``i``: e a mesma
    # checagem de h=1, deslocada pelo horizonte.
    origin_alignment = np.abs(
        frame["persistence"].to_numpy(float)[horizon:]
        - frame["actual"].to_numpy(float)[:-horizon]
    )
    if np.any(origin_alignment > 1e-8):
        raise ValueError("forecast origins do not align with prior realized prices")

    weekly_liters = float(monthly_liters * 12.0 / 52.0)
    shifted_liters = weekly_liters * float(flexibility_fraction)
    savings = np.zeros(len(frame) - horizon, dtype=float)
    events: list[dict[str, object]] = []
    correct_triggers = 0
    for index in range(len(frame) - horizon):
        current = float(frame["actual"].iloc[index])
        next_actual = float(frame["actual"].iloc[index + horizon])
        next_origin = float(frame["persistence"].iloc[index + horizon])
        next_prediction = float(frame[prediction_column].iloc[index + horizon])
        predicted_change = next_prediction - next_origin
        if predicted_change < signal_threshold_brl_per_liter or shifted_liters == 0:
            continue
        actual_change = next_actual - current
        gross = shifted_liters * actual_change
        # Carregar o volume por ``h`` semanas custa ``h`` vezes o carrego semanal.
        carrying = shifted_liters * carrying_cost_brl_per_liter_week * horizon
        net = gross - carrying
        savings[index] = net
        correct = actual_change > 0
        correct_triggers += int(correct)
        events.append(
            {
                "decision_date": str(frame["target_date"].iloc[index].date()),
                "consumption_date": str(frame["target_date"].iloc[index + horizon].date()),
                "horizon_weeks": horizon,
                "known_price": current,
                "forecast_price": next_prediction,
                "realized_next_price": next_actual,
                "predicted_change": predicted_change,
                "actual_change": actual_change,
                "liters_shifted": shifted_liters,
                "gross_savings_brl": gross,
                "carrying_cost_brl": carrying,
                "net_savings_brl": net,
                "direction_correct": correct,
            }
        )

    baseline_cost = weekly_liters * float(frame["actual"].sum())
    total_savings = float(savings.sum())
    largest_event = max(0.0, float(np.max(savings))) if savings.size else 0.0
    consumed_liters = weekly_liters * len(frame)
    ci = _annualized_block_bootstrap_ci(savings, random_state=random_state, horizon=horizon)
    triggered = len(events)
    return ProcurementBacktest(
        model=model_name,
        period_start=str(frame["target_date"].iloc[0].date()),
        period_end=str(frame["target_date"].iloc[-1].date()),
        n_weeks=len(frame),
        horizon_weeks=horizon,
        monthly_liters=float(monthly_liters),
        weekly_liters=weekly_liters,
        flexibility_fraction=float(flexibility_fraction),
        signal_threshold_brl_per_liter=float(signal_threshold_brl_per_liter),
        carrying_cost_brl_per_liter_week=float(carrying_cost_brl_per_liter_week),
        triggered_prebuys=triggered,
        trigger_precision=(correct_triggers / triggered) if triggered else None,
        liters_shifted=float(triggered * shifted_liters),
        baseline_cost_brl=baseline_cost,
        policy_cost_brl=baseline_cost - total_savings,
        net_savings_brl=total_savings,
        savings_excluding_largest_event_brl=total_savings - largest_event,
        largest_event_share_of_savings=(
            largest_event / total_savings if total_savings > 0 else None
        ),
        savings_per_liter_consumed_brl=total_savings / consumed_liters,
        annualized_savings_brl=float(np.mean(savings) * 52.0),
        annualized_savings_ci90_brl=ci,
        weekly_net_savings_brl=tuple(float(value) for value in savings),
        events=tuple(events),
    )


def simulate_one_week_prebuy(
    predictions: pd.DataFrame,
    *,
    prediction_column: str = "arima",
    model_name: str = "ARIMA",
    monthly_liters: float = 200_000.0,
    flexibility_fraction: float = 0.25,
    signal_threshold_brl_per_liter: float = 0.01,
    carrying_cost_brl_per_liter_week: float = 0.0,
    random_state: int = 42,
) -> ProcurementBacktest:
    """Politica de antecipacao de uma semana: o caso ``horizon=1``.

    Mantida como entrada propria porque e a politica dos backtests de h=1 ja
    publicados, e os numeros que ela devolve nao mudaram.
    """

    return simulate_horizon_prebuy(
        predictions,
        horizon=1,
        prediction_column=prediction_column,
        model_name=model_name,
        monthly_liters=monthly_liters,
        flexibility_fraction=flexibility_fraction,
        signal_threshold_brl_per_liter=signal_threshold_brl_per_liter,
        carrying_cost_brl_per_liter_week=carrying_cost_brl_per_liter_week,
        random_state=random_state,
    )
