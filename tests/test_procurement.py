from __future__ import annotations

import pandas as pd
import pytest

from vs_epl_krls.procurement import simulate_horizon_prebuy, simulate_one_week_prebuy


def _predictions() -> pd.DataFrame:
    actual = [6.00, 6.10, 6.05, 6.20, 6.25, 6.20, 6.30, 6.35, 6.40, 6.45]
    persistence = [5.95, *actual[:-1]]
    forecast = [6.00, 6.08, 6.08, 6.15, 6.24, 6.24, 6.28, 6.34, 6.42, 6.47]
    return pd.DataFrame(
        {
            "target_date": pd.date_range("2026-01-04", periods=len(actual), freq="7D"),
            "actual": actual,
            "persistence": persistence,
            "arima": forecast,
        }
    )


def test_procurement_replay_is_causal_balanced_and_deterministic():
    first = simulate_one_week_prebuy(
        _predictions(), monthly_liters=200_000, flexibility_fraction=0.25
    )
    second = simulate_one_week_prebuy(
        _predictions(), monthly_liters=200_000, flexibility_fraction=0.25
    )
    assert first.as_dict() == second.as_dict()
    assert first.triggered_prebuys > 0
    assert first.policy_cost_brl == pytest.approx(
        first.baseline_cost_brl - first.net_savings_brl
    )
    assert first.liters_shifted == pytest.approx(
        first.triggered_prebuys * first.weekly_liters * 0.25
    )
    assert first.annualized_savings_ci90_brl[0] <= first.annualized_savings_brl <= first.annualized_savings_ci90_brl[1]


def test_carrying_cost_reduces_measured_savings():
    zero = simulate_one_week_prebuy(_predictions(), carrying_cost_brl_per_liter_week=0)
    costly = simulate_one_week_prebuy(_predictions(), carrying_cost_brl_per_liter_week=0.01)
    assert costly.net_savings_brl < zero.net_savings_brl


def test_procurement_replay_rejects_leakage_prone_or_invalid_frames():
    broken = _predictions()
    broken.loc[2, "persistence"] = 99
    with pytest.raises(ValueError, match="origins"):
        simulate_one_week_prebuy(broken)
    gap = _predictions().drop(index=3).reset_index(drop=True)
    with pytest.raises(ValueError, match="uninterrupted"):
        simulate_one_week_prebuy(gap)
    with pytest.raises(ValueError, match="monthly_liters"):
        simulate_one_week_prebuy(_predictions(), monthly_liters=0)



def _horizon_predictions(horizon: int) -> pd.DataFrame:
    """Painel onde a origem da linha ``i+h`` e o preco realizado na linha ``i``."""

    actual = [6.00, 6.10, 6.05, 6.20, 6.25, 6.20, 6.30, 6.35, 6.40, 6.45, 6.50, 6.44]
    persistence = [5.90] * horizon + actual[:-horizon]
    forecast = [value + 0.02 for value in actual]
    return pd.DataFrame(
        {
            "target_date": pd.date_range("2026-01-04", periods=len(actual), freq="7D"),
            "actual": actual,
            "persistence": persistence,
            "arima": forecast,
        }
    )


def test_horizon_one_reproduces_the_published_one_week_policy():
    frame = _predictions()
    assert (
        simulate_horizon_prebuy(frame, horizon=1).as_dict()
        == simulate_one_week_prebuy(frame).as_dict()
    )


def test_horizon_replay_shifts_decision_and_consumption_by_the_horizon():
    result = simulate_horizon_prebuy(_horizon_predictions(4), horizon=4)
    assert result.horizon_weeks == 4
    for event in result.events:
        decided = pd.Timestamp(str(event["decision_date"]))
        consumed = pd.Timestamp(str(event["consumption_date"]))
        assert consumed - decided == pd.Timedelta(weeks=4)
        assert event["horizon_weeks"] == 4


def test_carrying_cost_scales_with_the_horizon():
    """Antecipar quatro semanas paga quatro semanas de estoque, nao uma."""

    free = simulate_horizon_prebuy(_horizon_predictions(4), horizon=4)
    charged = simulate_horizon_prebuy(
        _horizon_predictions(4), horizon=4, carrying_cost_brl_per_liter_week=0.01
    )
    assert charged.events, "o caso de teste precisa disparar ao menos uma compra"
    for event in charged.events:
        assert event["carrying_cost_brl"] == pytest.approx(
            event["liters_shifted"] * 0.01 * 4
        )
    assert charged.net_savings_brl < free.net_savings_brl


def test_horizon_replay_rejects_misaligned_origins():
    frame = _horizon_predictions(4)
    frame.loc[6, "persistence"] = frame.loc[6, "persistence"] + 0.5
    with pytest.raises(ValueError, match="origins do not align"):
        simulate_horizon_prebuy(frame, horizon=4)


def test_horizon_must_be_positive_and_shorter_than_the_panel():
    with pytest.raises(ValueError, match="horizon must be positive"):
        simulate_horizon_prebuy(_predictions(), horizon=0)
    with pytest.raises(ValueError, match="longer than the horizon"):
        simulate_horizon_prebuy(_predictions(), horizon=len(_predictions()))
