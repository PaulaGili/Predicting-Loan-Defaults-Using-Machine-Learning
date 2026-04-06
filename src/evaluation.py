
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

log = logging.getLogger("loan_default.evaluation")


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "roc_auc":   roc_auc,
        "threshold": threshold,
        "precision": report["1"]["precision"],
        "recall":    report["1"]["recall"],
        "f1":        report["1"]["f1-score"],
        "accuracy":  report["accuracy"],
    }

    log.info(
        f"{model_name} (t={threshold:.2f}) | AUC: {roc_auc:.4f} | "
        f"P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}"
    )
    return metrics


def plot_roc_curves(models: dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random")

    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")

    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves — Loan Default Models")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(output_dir) / "roc_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Saved {out}")


def plot_pr_curves(models: dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    baseline = float(y_test.mean())
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1.2, label=f"No-skill ({baseline:.2f})")

    for name, model in models.items():
        prec, rec, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(rec, prec, lw=2, label=f"{name} (AUC={auc(rec, prec):.3f})")

    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curves — Loan Default Models")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(output_dir) / "pr_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Saved {out}")


def plot_confusion_matrices(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
    thresholds: dict[str, float] | None = None,
) -> None:
    labels = ["Fully Paid", "Charged Off"]
    thresholds = thresholds or {}

    for name, model in models.items():
        t = thresholds.get(name, 0.5)
        y_pred = (model.predict_proba(X_test)[:, 1] >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
        ax.set(title=f"{name} (threshold={t:.2f})", xlabel="Predicted", ylabel="Actual")
        fig.tight_layout()

        out = Path(output_dir) / f"confusion_matrix_{name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        log.info(f"Saved {out}")


def plot_shap_summary(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    model_name: str,
    output_dir: str,
    n_background: int = 500,
    n_explain: int = 500,
) -> None:
    log.info(f"Computing SHAP for {model_name}...")
    try:
        rng = np.random.default_rng(42)
        bg   = X_train[rng.choice(len(X_train), min(n_background, len(X_train)), replace=False)]
        X_ex = X_test[rng.choice(len(X_test),   min(n_explain,     len(X_test)),  replace=False)]

        explainer = shap.Explainer(model, bg, feature_names=feature_names)
        shap_vals = explainer(X_ex, check_additivity=False)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, X_ex, feature_names=feature_names, show=False)
        plt.title(f"SHAP Feature Importance — {model_name}", fontsize=14)
        plt.tight_layout()
        out = Path(output_dir) / f"shap_summary_{model_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved {out}")

        top3 = np.argsort(np.abs(shap_vals.values).mean(axis=0))[::-1][:3]
        for idx in top3:
            feat = feature_names[idx]
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(idx, shap_vals.values, X_ex,
                                 feature_names=feature_names, ax=ax, show=False)
            ax.set_title(f"SHAP Dependence: {feat}", fontsize=13)
            fig.tight_layout()
            safe = feat.replace("/", "_").replace(" ", "_")
            dep_out = Path(output_dir) / f"shap_dep_{safe}.png"
            fig.savefig(dep_out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(f"Saved {dep_out}")

    except Exception as e:
        log.warning(f"SHAP failed for {model_name}: {e}")


def save_metrics_report(
    metrics_default: dict[str, dict[str, float]],
    metrics_tuned: dict[str, dict[str, float]],
    output_dir: str,
) -> None:
    df_def = pd.DataFrame(metrics_default).T.add_suffix("_t0.5")
    df_opt = pd.DataFrame(metrics_tuned).T.add_suffix("_optimal")
    df = pd.concat([df_def, df_opt], axis=1)
    df.index.name = "model"

    out = Path(output_dir) / "metrics_summary.csv"
    df.to_csv(out)
    log.info(f"Saved {out}")

    cols = ["roc_auc_t0.5", "f1_t0.5", "recall_t0.5", "f1_optimal", "recall_optimal"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(df[cols].round(4).to_string())
    print("=" * 70 + "\n")
