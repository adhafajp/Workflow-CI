"""
Output: X_train.csv, X_test.csv, y_train.csv, y_test.csv
"""

import os
import joblib
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.model_selection import train_test_split
from preprocessing_utils import clip_age, clip_emplen, clip_loanpct, log1p_clip # fungsi tranformasi custom

sklearn.set_config(transform_output="pandas")

# =====================
# LOGGING CONFIGURATION
# =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============
# FUNGSI UTAMA
# ============
def load_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data load successfully. Shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame, target_col: str = "loan_status"):
    logger.info("Splitting data into train and test sets BEFORE preprocessing...")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    logger.info("Building and applying preprocessing pipeline...")

    train_medians = X_train.groupby("loan_grade")["loan_int_rate"].median()

    X_train["loan_int_rate"] = X_train.groupby("loan_grade")["loan_int_rate"].transform(
        lambda x: x.fillna(x.median())
    )
    X_test["loan_int_rate"] = X_test.apply(
        lambda row: (
            train_medians.get(row["loan_grade"], X_train["loan_int_rate"].median())
            if pd.isna(row["loan_int_rate"])
            else row["loan_int_rate"]
        ),
        axis=1,
    )

    # Pipeline
    age_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clip", FunctionTransformer(clip_age, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )

    emplen_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clip", FunctionTransformer(clip_emplen, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )

    loanpct_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clip", FunctionTransformer(clip_loanpct, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )

    income_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("logclip", FunctionTransformer(log1p_clip, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
            ),
        ]
    )

    grade_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(categories=[["A", "B", "C", "D", "E", "F", "G"]]),
            ),
        ]
    )

    rate_pipe = Pipeline([("scaler", StandardScaler())])

    num_cols = ["loan_amnt", "cb_person_cred_hist_length"]
    num_pipe = Pipeline([("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        [
            ("age", age_pipe, ["person_age"]),
            ("emplen", emplen_pipe, ["person_emp_length"]),
            ("loanpct", loanpct_pipe, ["loan_percent_income"]),
            ("income", income_pipe, ["person_income"]),
            ("num", num_pipe, num_cols),
            ("rate", rate_pipe, ["loan_int_rate"]),
            (
                "cat",
                cat_pipe,
                ["person_home_ownership", "loan_intent", "cb_person_default_on_file"],
            ),
            ("grade", grade_pipe, ["loan_grade"]),
        ],
        remainder="drop",
    )

    logger.info("Fitting pipeline on X_train...")
    X_train_processed = preprocessor.fit_transform(X_train)

    logger.info("Transforming X_test...")
    X_test_processed = preprocessor.transform(X_test)

    clean_columns = [col.split("__")[-1] for col in X_train_processed.columns]
    X_train_processed.columns = clean_columns
    X_test_processed.columns = clean_columns

    logger.info(f"Preprocessing complete. Total feature: {X_train_processed.shape[1]}")
    return X_train_processed, X_test_processed, preprocessor


def save_output(X_train, X_test, y_train, y_test, output_dir: str, preprocessor=None):
    logger.info(f"Saving output to: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    if preprocessor is not None:
        joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))
        logger.info("Preprocessor saved as preprocessor.pkl")
    logger.info("All files saved successfuly")


def main():
    script_dir = Path(__file__).parent.absolute()
    input_file = script_dir  / "credit_risk_preprocessing/raw/credit_risk_dataset.csv"
    output_dir = script_dir / "credit_risk_preprocessing"

    logger.info("=" * 60)
    logger.info("AUTOMATE PREPROCESSING - CREDIT RISK DATASET")
    logger.info("=" * 60)

    df = load_data(str(input_file))
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_processed, X_test_processed, preprocessor  = preprocess_data(X_train, X_test)
    save_output(X_train_processed, X_test_processed, y_train, y_test, str(output_dir), preprocessor)


# ====
# RUN
# ====
if __name__ == "__main__":
    main()
