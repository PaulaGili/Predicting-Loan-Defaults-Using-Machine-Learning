import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.model_training import (
    _instantiate,
    find_best_threshold,
    get_model_definitions,
    train_final_model,
)


@pytest.fixture
def synthetic_data():
    X, y = make_classification(
        n_samples=300, n_features=10, weights=[0.82, 0.18], random_state=42
    )
    return X.astype(float), y


def test_find_best_threshold_in_range(synthetic_data):
    X, y = synthetic_data
    model = _instantiate("LogisticRegression", {"C": 1.0}, seed=42)
    model.fit(X, y)
    t = find_best_threshold(model, X, y)
    assert 0.0 < t < 1.0


def test_get_model_definitions_respects_enabled_flag(minimal_config):
    models = get_model_definitions(minimal_config)
    assert "LightGBM" in models
    assert "RandomForest" not in models
    assert "LogisticRegression" not in models


def test_instantiate_all_models():
    cases = [
        ("LogisticRegression",   {"C": 1.0}),
        ("RandomForest",         {"n_estimators": 10}),
        ("XGBoost",              {"n_estimators": 10}),
        ("LightGBM",             {"n_estimators": 10}),
        ("HistGradientBoosting", {"max_iter": 10}),
    ]
    for name, params in cases:
        model = _instantiate(name, params, seed=42)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")


def test_train_final_model_lgbm(synthetic_data, minimal_config):
    X, y = synthetic_data
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    model = train_final_model(
        "LightGBM",
        best_params={"n_estimators": 20, "scale_pos_weight": 4.5},
        X_train=X_df,
        y_train=y,
        config=minimal_config,
    )
    proba = model.predict_proba(X_df.iloc[:5])
    assert proba.shape == (5, 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_train_final_model_rf_uses_smote(synthetic_data, minimal_config):
    X, y = synthetic_data
    model = train_final_model(
        "RandomForest",
        best_params={"n_estimators": 20},
        X_train=X,
        y_train=y,
        config=minimal_config,
    )
    assert model.predict(X[:5]).shape == (5,)
