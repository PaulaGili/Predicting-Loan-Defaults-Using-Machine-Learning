
import logging
from typing import Any

import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

log = logging.getLogger("loan_default.training")

# LightGBM and XGBoost handle imbalance via scale_pos_weight, so we skip SMOTE for them
_NATIVE_IMBALANCE = frozenset({"LightGBM", "XGBoost"})


def get_model_definitions(config: dict[str, Any]) -> dict[str, Any]:
    seed = config["training"]["random_seed"]
    mc = config.get("models", {})
    models: dict[str, Any] = {}

    if mc.get("logistic_regression", {}).get("enabled", True):
        models["LogisticRegression"] = LogisticRegression(
            random_state=seed,
            max_iter=mc.get("logistic_regression", {}).get("max_iter", 2000),
            solver="liblinear",
        )
    if mc.get("random_forest", {}).get("enabled", True):
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        )
    if mc.get("xgboost", {}).get("enabled", True):
        models["XGBoost"] = XGBClassifier(random_state=seed, verbosity=0)
    if mc.get("lightgbm", {}).get("enabled", True):
        models["LightGBM"] = LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1)
    if mc.get("hist_gradient_boosting", {}).get("enabled", True):
        models["HistGradientBoosting"] = HistGradientBoostingClassifier(random_state=seed)

    log.info(f"Models: {list(models.keys())}")
    return models


def build_stacking_model(
    config: dict[str, Any],
    tuned_params: dict[str, Any] | None = None,
) -> StackingClassifier:
    seed = config["training"]["random_seed"]

    def _base_params(name: str) -> dict:
        if tuned_params is None:
            return {}
        params = dict(tuned_params.get(name, {}))
        # SMOTE is applied outside before stacking.fit(), so drop scale_pos_weight
        params.pop("scale_pos_weight", None)
        return params

    estimators = [
        ("rf",   _instantiate("RandomForest", _base_params("RandomForest"), seed)),
        ("xgb",  _instantiate("XGBoost",      _base_params("XGBoost"),      seed)),
        ("lgbm", _instantiate("LightGBM",      _base_params("LightGBM"),    seed)),
    ]

    # passthrough=True gives the meta-learner the raw features too, not just the 3 probas
    # StandardScaler is needed because LogReg is sensitive to feature scale
    final_estimator = SkPipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=0.1, max_iter=1000, random_state=seed)),
    ])

    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=1,  # nested parallelism with LightGBM on Windows causes issues
    )


def find_best_threshold(model: Any, X_val: np.ndarray, y_val: np.ndarray) -> float:
    # 0.5 is basically arbitrary for imbalanced data, find the cutoff that maximizes F1
    y_prob = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    return float(thresholds[int(np.argmax(f1))])


def _build_objective(model_name: str, X: np.ndarray, y: np.ndarray, config: dict[str, Any]):
    seed = config["training"]["random_seed"]
    n_splits = config["training"]["n_splits"]

    def objective(trial: optuna.Trial) -> float:
        if model_name == "LogisticRegression":
            params: dict[str, Any] = {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            }
        elif model_name == "RandomForest":
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 300, step=50),
                "max_depth":        trial.suggest_int("max_depth", 4, 12),
                "min_samples_split":trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
        elif model_name == "XGBoost":
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 800, step=50),
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 3.0, 8.0),
            }
        elif model_name == "LightGBM":
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 200, 1000, step=50),
                "max_depth":         trial.suggest_int("max_depth", 3, 12),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 3.0, 8.0),
                "bagging_freq":      trial.suggest_int("bagging_freq", 1, 7),  # needed for subsample to work
            }
        elif model_name == "HistGradientBoosting":
            params = {
                "max_iter":          trial.suggest_int("max_iter", 100, 600, step=50),
                "max_depth":         trial.suggest_int("max_depth", 3, 12),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 50),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 1.0, log=True),
            }
        else:
            raise ValueError(f"No search space defined for {model_name}")

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            if model_name not in _NATIVE_IMBALANCE:
                X_tr, y_tr = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)

            model = _instantiate(model_name, params, seed)
            model.fit(X_tr, y_tr)
            scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

        return float(np.mean(scores))

    return objective


def _instantiate(model_name: str, tuned_params: dict[str, Any], seed: int) -> Any:
    if model_name == "LogisticRegression":
        return LogisticRegression(**tuned_params, solver="liblinear", random_state=seed, max_iter=2000)
    if model_name == "RandomForest":
        return RandomForestClassifier(**tuned_params, random_state=seed, n_jobs=-1)
    if model_name == "XGBoost":
        return XGBClassifier(**tuned_params, random_state=seed, verbosity=0)
    if model_name == "LightGBM":
        return LGBMClassifier(**tuned_params, random_state=seed, verbose=-1, n_jobs=-1)
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(**tuned_params, random_state=seed)
    raise ValueError(f"Unknown model: {model_name}")


def tune_model(model_name: str, X: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    n_trials = config["optuna"]["n_trials"]
    timeout = config["optuna"].get("timeout")
    log.info(f"Optuna: {model_name} | {n_trials} trials")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        _build_objective(model_name, X, y, config),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    log.info(f"{model_name} best CV AUC: {study.best_value:.4f} | {study.best_params}")
    return study.best_params


def train_final_model(
    model_name: str,
    best_params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
) -> Any:
    seed = config["training"]["random_seed"]
    model = _instantiate(model_name, best_params, seed)

    if model_name in _NATIVE_IMBALANCE:
        model.fit(X_train, y_train)
        log.info(f"Trained {model_name} (scale_pos_weight={best_params.get('scale_pos_weight', 'n/a')})")
    else:
        X_res, y_res = SMOTE(random_state=seed).fit_resample(X_train, y_train)
        log.info(f"SMOTE: {len(X_train):,} -> {len(X_res):,} samples")
        model.fit(X_res, y_res)
        log.info(f"Trained {model_name}")

    return model
