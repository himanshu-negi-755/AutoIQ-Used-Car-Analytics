"""Tests for the price model pipeline and inference helpers."""
import numpy as np
import pandas as pd
import pytest

from data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from model import get_model_data
from model import FEATURES, build_pipeline, candidate_models, feature_importance, predict_price, split_data


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return get_model_data().sample(800, random_state=0)


@pytest.fixture(scope="module")
def fitted_rf(data):
    from sklearn.ensemble import RandomForestRegressor
    pipe = build_pipeline(RandomForestRegressor(n_estimators=30, random_state=0, n_jobs=-1))
    X_train, X_test, y_train, y_test = split_data(data)
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test


def test_feature_lists_are_disjoint_and_complete():
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert set(FEATURES) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)


def test_split_uses_log_target(data):
    _, _, y_train, _ = split_data(data)
    assert y_train.max() < 20          # log1p of 1e7 ≈ 16


def test_candidate_models_have_four_families():
    assert len(candidate_models()) == 4


def test_pipeline_fits_and_generalises(fitted_rf):
    pipe, X_test, y_test = fitted_rf
    from sklearn.metrics import r2_score
    assert r2_score(y_test, pipe.predict(X_test)) > 0.75


def test_unknown_brand_does_not_crash(fitted_rf):
    pipe, X_test, _ = fitted_rf
    row = X_test.iloc[[0]].copy()
    row["brand"] = "Porsche"
    assert np.isfinite(pipe.predict(row)[0])


def test_predict_price_returns_band(fitted_rf):
    pipe, X_test, _ = fitted_rf
    car = X_test.iloc[0].to_dict()
    res = predict_price(pipe, car, mape=10)
    assert res["low"] < res["predicted_price"] < res["high"]
    assert res["low"] == pytest.approx(res["predicted_price"] * 0.9, rel=0.01)


def test_older_car_predicts_cheaper(fitted_rf):
    pipe, X_test, _ = fitted_rf
    car = X_test.iloc[0].to_dict()
    young = predict_price(pipe, {**car, "car_age": 2})["predicted_price"]
    old = predict_price(pipe, {**car, "car_age": 12})["predicted_price"]
    assert old < young


def test_feature_importance_aggregates_categoricals(fitted_rf):
    pipe, _, _ = fitted_rf
    imp = feature_importance(pipe)
    assert set(imp["feature"]) == set(FEATURES)
    assert imp["importance"].sum() == pytest.approx(1.0, abs=1e-6)
