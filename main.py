"""
Loan Default Prediction Pipeline

Usage:
    python main.py                          # full run with Optuna tuning
    python main.py --no-tune                # skip tuning, much faster
    python main.py --trials 20              # fewer Optuna trials
    mlflow ui                               # view experiment history
"""


import argparse
from pathlib import Path

import warnings

import joblib
import mlflow
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_data, preprocess
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrices,
    plot_pr_curves,
    plot_roc_curves,
    plot_shap_summary,
    save_metrics_report,
)
from src.feature_engineering import build_feature_pipeline
from src.model_training import (
    build_stacking_model,
    find_best_threshold,
    get_model_definitions,
    train_final_model,
    tune_model,
)
from src.utils import ensure_dirs, load_config, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Loan Default Prediction Pipeline")
    p.add_argument("--config",  default="config/config.yaml")
    p.add_argument("--no-tune", action="store_true", help="skip Optuna, use default params")
    p.add_argument("--trials",  type=int, default=None, help="override n_trials in config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.trials is not None:
        config["optuna"]["n_trials"] = args.trials

    ensure_dirs(config)
    log = setup_logging(config["paths"]["logs"])

    seed: int = config["training"]["random_seed"]
    output_dir: str = config["paths"]["outputs"]
    models_dir: str = config["paths"]["models"]

    tuning_str = "disabled" if args.no_tune else f"Optuna ({config['optuna']['n_trials']} trials)"
    log.info(f"Starting pipeline | tuning: {tuning_str}")

    mlflow.set_experiment("loan-default-prediction")
    run = mlflow.start_run()
    log.info(f"MLflow run: {run.info.run_id}")

    try:
        mlflow.log_params({
            "test_size":   config["training"]["test_size"],
            "random_seed": seed,
            "n_trials":    config["optuna"]["n_trials"],
            "no_tune":     args.no_tune,
            "n_splits":    config["training"]["n_splits"],
        })

        log.info("Step 1 — Load and preprocess")
        df = load_data(config["data"]["raw_path"])
        X_raw, y = preprocess(df, config)
        mlflow.log_params({
            "n_samples":    len(X_raw),
            "default_rate": round(float(y.mean()), 4),
        })

        log.info("Step 2 — Train/test split")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y,
            test_size=config["training"]["test_size"],
            random_state=seed,
            stratify=y,
        )
        log.info(f"Train: {len(X_train_raw):,} | Test: {len(X_test_raw):,}")

        log.info("Step 3 — Feature engineering")
        feat_pipeline = build_feature_pipeline(config)
        X_train_df = feat_pipeline.fit_transform(X_train_raw)
        X_test_df  = feat_pipeline.transform(X_test_raw)

        feature_names: list[str] = list(X_train_df.columns)
        X_train = X_train_df.values.astype(float)
        X_test  = X_test_df.values.astype(float)

        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        y_train_arr = y_train.values
        y_test_arr  = y_test.values
        log.info(f"Features: {len(feature_names)} | Train: {X_train.shape} | Test: {X_test.shape}")
        mlflow.log_param("n_features", len(feature_names))

        log.info("Step 4 — Model training")
        model_defs = get_model_definitions(config)
        trained_models: dict = {}
        best_params_per_model: dict = {}

        for model_name, default_model in model_defs.items():
            log.info(f"  [{model_name}]")
            if args.no_tune:
                X_res, y_res = SMOTE(random_state=seed).fit_resample(X_train, y_train_arr)
                default_model.fit(X_res, y_res)
                trained_models[model_name] = default_model
            else:
                best_params = tune_model(model_name, X_train, y_train_arr, config)
                best_params_per_model[model_name] = best_params
                trained_models[model_name] = train_final_model(
                    model_name, best_params, X_train, y_train_arr, config
                )
                mlflow.log_params({f"{model_name}_{k}": v for k, v in best_params.items()})

        if config.get("models", {}).get("stacking", {}).get("enabled", True):
            log.info("Step 5 — Stacking Ensemble")
            tuned = best_params_per_model if not args.no_tune else None
            stacking = build_stacking_model(config, tuned_params=tuned)
            X_res, y_res = SMOTE(random_state=seed).fit_resample(X_train, y_train_arr)
            stacking.fit(X_res, y_res)
            trained_models["Stacking"] = stacking

        log.info("Step 6 — Evaluation")
        metrics_default: dict = {}
        metrics_tuned: dict = {}
        optimal_thresholds: dict = {}

        for name, model in trained_models.items():
            metrics_default[name] = evaluate_model(model, X_test, y_test_arr, name, threshold=0.5)
            t = find_best_threshold(model, X_test, y_test_arr)
            optimal_thresholds[name] = t
            metrics_tuned[name] = evaluate_model(model, X_test, y_test_arr, f"{name}*", threshold=t)
            mlflow.log_metrics({
                f"{name}_auc":       metrics_default[name]["roc_auc"],
                f"{name}_f1":        metrics_tuned[name]["f1"],
                f"{name}_precision": metrics_tuned[name]["precision"],
                f"{name}_recall":    metrics_tuned[name]["recall"],
                f"{name}_threshold": t,
            })

        best_name = max(metrics_default, key=lambda k: metrics_default[k]["roc_auc"])
        best_model = trained_models[best_name]
        log.info(f"Best model: {best_name} (AUC={metrics_default[best_name]['roc_auc']:.4f})")

        mlflow.log_params({"best_model": best_name})
        mlflow.log_metrics({
            "best_auc":       metrics_default[best_name]["roc_auc"],
            "best_f1":        metrics_tuned[best_name]["f1"],
            "best_threshold": optimal_thresholds[best_name],
        })

        log.info("Step 7 — Plots")
        plot_roc_curves(trained_models, X_test, y_test_arr, output_dir)
        plot_pr_curves(trained_models, X_test, y_test_arr, output_dir)
        plot_confusion_matrices(trained_models, X_test, y_test_arr, output_dir, optimal_thresholds)
        save_metrics_report(metrics_default, metrics_tuned, output_dir)
        plot_shap_summary(best_model, X_train, X_test, feature_names, best_name, output_dir)
        mlflow.log_artifacts(output_dir, artifact_path="plots")

        log.info("Step 8 — Saving best model")
        model_path = Path(models_dir) / f"{best_name}_best.joblib"
        joblib.dump(
            {
                "model":             best_model,
                "feature_pipeline":  feat_pipeline,
                "feature_names":     feature_names,
                "optimal_threshold": optimal_thresholds[best_name],
                "metrics":           metrics_tuned[best_name],
            },
            model_path,
        )
        mlflow.log_artifact(str(model_path))
        log.info(f"Saved -> {model_path}")
        log.info("Done.")

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
