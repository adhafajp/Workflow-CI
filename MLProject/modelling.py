import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    classification_report,
)
import mlflow
import mlflow.sklearn
import dagshub
import json
import joblib
from datetime import datetime

# ==============
# PARSE ARGUMENT
# ==============
parser = argparse.ArgumentParser()
parser.add_argument("--experiment-name", type=str, default="credit_risk_ci")
parser.add_argument("--run-name", type=str, default="ci_training_run")
args = parser.parse_args()

# =============
# CONFIGURATION
# =============
DAGSHUB_USER = os.environ.get("DAGSHUB_USER", "adhafajp")
DAGSHUB_REPO = os.environ.get("DAGSHUB_REPO", "credit_risk_model")

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"CI/CD mode - Using tracking URI: {tracking_uri}")
else:
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
    print("Local mode - Using DagsHub init")

mlflow.set_experiment(args.experiment_name)

# =========
# LOAD DATA
# =========
DATA_DIR = "credit_risk_preprocessing"

print("Loading data...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ===================================
# TRAINING WITH HYPERPARAMETER TUNING
# ===================================
with mlflow.start_run(run_name=args.run_name) as run:

    print("Starting hyperparameter tuning...")

    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    mlflow.log_params(grid_search.best_params_)

    # ==========
    # EVALUATION
    # ==========
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    mlflow.log_metrics(
        {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "best_cv_score": grid_search.best_score_,
        }
    )

    # =========
    # ARTIFACTS
    # =========
    print("Generating artifact...")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Non-Default", "Default"],
        yticklabels=["Non-Default", "Default"],
    )
    ax.set_title(f"Confusion Matrix\nAcc={acc:.4f} | ROC-AUC={roc_auc:.4f}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "darkorange", lw=2, label=f"AUC = {auc(fpr, tpr):.4f}")
    ax.plot([0, 1], [0, 1], "navy", lw=2, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    mlflow.log_artifact("roc_curve.png")
    plt.close()

    # Feature Importance
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(indices)), importances[indices])
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([X_train.columns[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Top 15 Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    # Metrics JSON
    metrics_dict = {
        "run_id": run.info.run_id,
        "timestamp": datetime.now().isoformat(),
        "best_parameters": grid_search.best_params_,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "cv_score": grid_search.best_score_,
        },
        "dataset_info": {
            "train_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
        },
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
    mlflow.log_artifact("metrics.json")

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact("classification_report.json")

    # =========
    # LOG MODEL
    # =========
    print("Logging model...")
    signature = mlflow.models.infer_signature(
        X_train.head(5), best_model.predict(X_train.head(5))
    )

    mlflow.sklearn.log_model(
        best_model,
        "model",
        signature=signature,
        registered_model_name="credit_risk_model",
        input_example=X_train.head(3),
    )

    # =======
    # SUMMARY
    # =======
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Run ID:      {run.info.run_id}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"Best Params: {grid_search.best_params_}")
    print("=" * 50)

    with open("run_info.txt", "w") as f:
        f.write(f"Run ID: {run.info.run_id}\n")
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
