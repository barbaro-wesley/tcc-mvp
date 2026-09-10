"""Regression coverage for the calendar, feature geometry and serving audit."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vs_epl_krls.selection import (
    TemporalFold, build_s10_feature_frame, build_s10_supervised,
    candidate_grid, evaluate_temporal_fold,
)
from vs_epl_krls.utils import MinMaxScaler, RobustBoundedScaler
from vs_epl_krls.weekly import weekly_grid
from vs_epl_krls.passthrough import (
    PARITY_FEATURES, PassThroughECM, build_next_parity_row, build_parity_panel,
)


def history(n=200):
    rng = np.random.default_rng(77)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-05", periods=n, freq="7D"),
        "price": 5 + np.cumsum(rng.normal(0.002, 0.02, n)),
        "parity": 2 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
    })


@pytest.mark.parametrize("horizon", [1, 2, 4])
def test_labels_follow_calendar_even_when_feature_rows_are_missing(horizon):
    raw = history().drop(index=range(100, 108))
    supervised = build_s10_supervised(raw, horizon=horizon, feature_set="lags")
    assert np.all(supervised.target_dates - supervised.dates == np.timedelta64(7*horizon, "D"))
    observed = raw.set_index("date")["price"]
    assert np.allclose(supervised.target_price, observed.reindex(supervised.target_dates))
    assert supervised.price_history["price"].isna().sum() == 8
    features = build_s10_feature_frame(raw, feature_set="lags")
    gap_end = history().date.iloc[107]
    assert not features.date.between(gap_end, gap_end + pd.Timedelta(weeks=12)).any()


def test_delayed_learning_reveals_all_eligible_labels_after_gap(monkeypatch):
    import vs_epl_krls.selection as selection
    raw = history().drop(index=100)
    data = build_s10_supervised(raw, horizon=4, feature_set="price")
    start = int(np.flatnonzero(data.dates >= np.datetime64(raw.date.iloc[100]))[0])
    learned = []
    snapshots = []
    original_learn = selection.VSEPLKRLS.learn_one
    original_predict = selection.VSEPLKRLS.predict_one

    def learn(self, x, y):
        learned.append(float(y))
        return original_learn(self, x, y)

    def predict(self, x):
        snapshots.append(len(learned))
        return original_predict(self, x)

    monkeypatch.setattr(selection.VSEPLKRLS, "learn_one", learn)
    monkeypatch.setattr(selection.VSEPLKRLS, "predict_one", predict)
    candidate = replace(candidate_grid(horizon=4, n_random=0)[0], target_mode="delta")
    evaluate_temporal_fold(candidate, data, TemporalFold("gap", start, start+5))
    assert snapshots == [data.known_target_end(i) for i in range(start, start+5)]
    assert snapshots[0] > start - data.horizon + 1


def test_scaler_preserves_distinct_prices_beyond_training_maximum():
    train = np.linspace(3, 5, 100).reshape(-1, 1)
    future = np.linspace(6, 9, 52).reshape(-1, 1)
    assert np.unique(np.clip(MinMaxScaler().fit(train).transform(future), 0, 1)).size == 1
    scaler = RobustBoundedScaler().fit(train)
    encoded = scaler.transform(future)
    assert np.unique(encoded).size == 52
    assert ((encoded > 0) & (encoded < 1)).all()
    assert (np.diff(encoded[:, 0]) > 0).all()
    assert np.array_equal(encoded[:1], scaler.transform(future[:1]))
    constant = RobustBoundedScaler().fit(np.ones((20, 2)))
    assert np.isfinite(constant.transform(np.array([[0, 100]]))).all()


def test_scaler_contract_and_legacy_candidate_defaults():
    with pytest.raises(RuntimeError, match="fitted"):
        RobustBoundedScaler().transform([[1]])
    with pytest.raises(ValueError, match="finite"):
        RobustBoundedScaler().fit([[np.nan]])
    scaler = RobustBoundedScaler().fit([[1, 2], [2, 3]])
    with pytest.raises(ValueError, match="shape"):
        scaler.transform([[1]])
    with pytest.raises(ValueError, match="finite"):
        scaler.transform([[1, np.inf]])
    candidate = candidate_grid(horizon=1, n_random=0)[0]
    assert isinstance(candidate.make_feature_scaler(), RobustBoundedScaler)
    assert isinstance(replace(candidate, feature_scaling="minmax").make_feature_scaler(), MinMaxScaler)
    with pytest.raises(ValueError, match="feature_scaling"):
        replace(candidate, feature_scaling="invalid")


def test_next_parity_features_equal_historical_features_without_future_values():
    raw = history(320)
    cut = 300
    next_row = build_next_parity_row(raw.iloc[:cut])
    columns = [*PARITY_FEATURES, "volatility", "abs_cost_move", "origin_price"]
    expected = build_parity_panel(raw).iloc[cut]
    assert np.allclose(next_row[columns].astype(float), expected[columns].astype(float))
    assert pd.isna(next_row.price) and pd.isna(next_row.y)
    tampered = raw.copy()
    tampered.loc[cut:, ["price", "parity"]] *= 10
    changed = build_parity_panel(tampered).iloc[cut]
    assert np.allclose(next_row[columns].astype(float), changed[columns].astype(float))
    fitted = PassThroughECM(feature_names=PARITY_FEATURES).fit(build_parity_panel(raw.iloc[:cut]))
    live = fitted.forecast_row(next_row, origin_price=float(next_row.origin_price))
    replay = fitted.forecast_row(expected, origin_price=float(expected.origin_price))
    assert live.point == pytest.approx(replay.point, abs=1e-12)
    assert live.conditional_sigma == pytest.approx(replay.conditional_sigma, abs=1e-12)


def test_parity_gap_does_not_turn_multweek_change_into_weekly_target():
    raw = history(320)
    gap = raw.date.iloc[200]
    panel = build_parity_panel(raw.drop(index=200)).set_index("date")
    assert pd.isna(panel.loc[gap, "price"])
    assert pd.isna(panel.loc[gap + pd.Timedelta(weeks=1), "y"])


def test_weekly_calendar_rejects_off_grid_and_duplicate_dates():
    raw = history()
    raw.loc[99, "date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="weekly calendar"):
        weekly_grid(raw)
    with pytest.raises(ValueError, match="unique"):
        weekly_grid(pd.concat([history(), history().iloc[:1]]))


def test_new_scaler_survives_production_serialization(tmp_path):
    from vs_epl_krls.production import S10ProductionForecaster
    candidate = replace(candidate_grid(horizon=1, n_random=0)[0], target_mode="delta")
    fitted = S10ProductionForecaster(candidate).fit(history())
    before = fitted.predict_next()
    path = fitted.save(tmp_path / "new.joblib")
    restored = S10ProductionForecaster.load(path)
    assert isinstance(restored.x_scaler_, RobustBoundedScaler)
    assert restored.predict_next().as_dict() == before.as_dict()


def test_selection_defaults_to_development_and_rejects_stale_cache(load_script, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from vs_epl_krls.selection import S10_HOLDOUT_END, S10_HOLDOUT_START

    script = load_script("05_s10_model_selection.py")
    raw = history(400)
    raw["date"] = pd.date_range(end=S10_HOLDOUT_END, periods=len(raw), freq="7D")
    monkeypatch.setattr(script, "load_anp_fuel_csv", lambda *a, **kw: raw.copy())

    def baseline(data, start, end):
        assert data.target_dates[end-1] < np.datetime64(S10_HOLDOUT_START)
        return data.origin_price[start:end]

    monkeypatch.setattr(script, "_arima_holdout", baseline)
    monkeypatch.setattr(script, "_ridge_holdout", baseline)
    args = SimpleNamespace(data=tmp_path / "unused", output_dir=tmp_path, horizon=1,
                           n_random=0, random_state=1, validation_size=12, n_folds=1,
                           holdout_size=104, min_train_size=80, reuse_validation=False)
    manifest = script.run(args)
    assert manifest["holdout_evaluated"] is False
    assert not list(tmp_path.glob("holdout*"))
    args.reuse_validation = True
    assert script.run(args)["validation_fingerprint"] == manifest["validation_fingerprint"]
    raw.loc[40, "price"] += 0.2
    with pytest.raises(ValueError, match="different data, code or configuration"):
        script.run(args)


def test_selection_arima_uses_full_calendar_and_production_fit(load_script, monkeypatch):
    script = load_script("05_s10_model_selection.py")
    raw = history().drop(index=80)
    data = build_s10_supervised(raw)
    fitted_histories = []

    class FakeARIMA:
        def forecast(self, steps):
            return np.repeat(5.0, steps)

    def fit(prices):
        fitted_histories.append(prices.copy())
        return FakeARIMA()

    monkeypatch.setattr(script.S10ProductionForecaster, "_fit_arima", fit)
    result = script._arima_holdout(data, data.n_samples-3, data.n_samples)
    assert len(result) == len(fitted_histories) == 3
    assert all(np.isnan(values[80]) for values in fitted_histories)
    assert len(fitted_histories[-1]) == len(raw)
    assert fitted_histories[-1][0] == raw.price.iloc[0]
