import os
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
    fbeta_score,
    make_scorer,
)
from preprocessing_utils import clip_age, clip_emplen, clip_loanpct, log1p_clip
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.pyfunc
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

# ============
# PREPROCESSOR
# ============
preprocessor = joblib.load(os.path.join(DATA_DIR, "preprocessor.pkl"))


# =====================
# CUSTOM PYFUNC WRAPPER
# =====================
class CreditRiskModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, preprocessor, threshold):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.transform(df)

    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        X = self._preprocess(model_input)

        probability = self.model.predict_proba(X)[:, 1].tolist()
        prediction = [1 if p >= self.threshold else 0 for p in probability]

        return pd.DataFrame(
            {
                "predict": prediction,
                "prob_predict": probability,
            }
        )


# ===================================
# TRAINING WITH HYPERPARAMETER TUNING
# ===================================
with mlflow.start_run(run_name=args.run_name) as run:

    print("Starting hyperparameter tuning...")

    f2_scorer = make_scorer(fbeta_score, beta=2)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")

    mlflow.log_params(
        {
            "smote_strategy": "auto",
            "train_shape_before_smote": str(list(X_train.shape)),
            "train_shape_after_smote": str(list(X_train_balanced.shape)),
        }
    )

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring=f2_scorer, n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train_balanced, y_train_balanced)
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # ======================
    # THRESHOLD OPTIMIZATION
    # ======================
    print("Finding optimal threshold...")
    y_proba_test = best_model.predict_proba(X_test)[:, 1]

    best_threshold = 0.5
    best_recall = 0.0
    PRECISION_FLOOR = 0.65

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred_t = (y_proba_test >= threshold).astype(int)
        rec = recall_score(y_test, y_pred_t, zero_division=0)
        prec = precision_score(y_test, y_pred_t, zero_division=0)

        if rec > best_recall and prec >= PRECISION_FLOOR:
            best_recall = rec
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold:.2f} (recall: {best_recall:.4f})")
    mlflow.log_param("optimal_threshold", round(best_threshold, 2))

    # ==========
    # EVALUATION
    # ==========
    y_pred = (y_proba_test >= best_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba_test)

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
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
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
            "train_shape_original": list(X_train.shape),
            "train_shape_balanced": list(X_train_balanced.shape),
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
    RAW_COLS = {
        "person_age": "double",
        "person_income": "double",
        "person_emp_length": "double",
        "loan_grade": "string",
        "loan_amnt": "double",
        "loan_int_rate": "double",
        "loan_percent_income": "double",
        "cb_person_cred_hist_length": "double",
        "person_home_ownership": "string",
        "loan_intent": "string",
        "cb_person_default_on_file": "string",
    }

    input_schema = mlflow.types.Schema(
        [mlflow.types.ColSpec(dtype, col) for col, dtype in RAW_COLS.items()]
    )

    output_schema = mlflow.types.Schema(
        [
            mlflow.types.ColSpec("long", "predict"),
            mlflow.types.ColSpec("double", "prob_predict"),
        ]
    )

    signature = mlflow.models.ModelSignature(
        inputs=input_schema,
        outputs=output_schema,
    )

    raw_example = pd.DataFrame(
        [
            {
                "person_age": 28,
                "person_income": 45000,
                "person_emp_length": 3,
                "loan_grade": "C",
                "loan_amnt": 10000,
                "loan_int_rate": 12.5,
                "loan_percent_income": 0.22,
                "cb_person_cred_hist_length": 4,
                "person_home_ownership": "RENT",
                "loan_intent": "PERSONAL",
                "cb_person_default_on_file": "N",
            }
        ]
    )

    wrapped_model = CreditRiskModelWrapper(
        model=best_model,
        preprocessor=preprocessor,
        threshold=best_threshold,
    )

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapped_model,
        signature=signature,
        input_example=raw_example,
        registered_model_name="credit_risk_model",
        artifacts={"preprocessor": os.path.join(DATA_DIR, "preprocessor.pkl")},
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
