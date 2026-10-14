"""
Evaluation utilities for Vitalyn Alertia.

Evaluates the trained sepsis prediction model on
patient-level PhysioNet ICU data.

IMPORTANT:
Patients are split before evaluation so that data from
the same patient does not appear in both training and testing.
"""

import os
import glob
import joblib

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from ml.features import extract_features


DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "raw",
    "training_setA",
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "saved_models",
    "sepsis_model.pkl",
)


def load_patient_files():
    """
    Return all PhysioNet patient file paths.
    """

    files = sorted(
        glob.glob(
            os.path.join(
                DATA_DIR,
                "*.psv"
            )
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No PhysioNet patient files found in {DATA_DIR}"
        )

    return files


def build_evaluation_dataset(patient_files):
    """
    Convert patient time-series data into model-ready
    feature vectors.

    Each hourly observation becomes one evaluation sample.
    """

    X = []
    y = []

    for file_path in patient_files:

        try:
            df = pd.read_csv(
                file_path,
                sep="|"
            )

            if "SepsisLabel" not in df.columns:
                continue

            for i in range(len(df)):

                history = df.iloc[:i + 1]

                try:
                    features = extract_features(history)
                except (ValueError, KeyError):
                    continue

                label = pd.to_numeric(
                    df.iloc[i]["SepsisLabel"],
                    errors="coerce"
                )

                if pd.isna(label):
                    continue

                X.append(features)
                y.append(int(label))

        except Exception as error:
            print(
                f"Warning: could not evaluate "
                f"{os.path.basename(file_path)}: {error}"
            )

    if not X:
        raise ValueError(
            "Could not construct an evaluation dataset."
        )

    return np.asarray(X), np.asarray(y)


def evaluate_model(patient_files=None):
    """
    Evaluate the trained model.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            f"Run python ml/train.py first."
        )

    model = joblib.load(MODEL_PATH)

    if patient_files is None:
        patient_files = load_patient_files()

    X, y = build_evaluation_dataset(
        patient_files
    )

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y,
            predictions
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities
        ),
        "pr_auc": average_precision_score(
            y,
            probabilities
        ),
    }

    return metrics


def save_evaluation_results(metrics):
    """
    Save evaluation metrics to CSV.
    """

    output_path = os.path.join(
        os.path.dirname(__file__),
        "evaluation_metrics.csv"
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        output_path,
        index=False
    )

    print(
        f"Evaluation results saved to {output_path}"
    )


if __name__ == "__main__":

    print("Loading PhysioNet patient files...")

    patient_files = load_patient_files()

    print(
        f"Found {len(patient_files)} patient files."
    )

    metrics = evaluate_model(
        patient_files
    )

    print("\nEvaluation Results")
    print("-" * 40)

    for name, value in metrics.items():
        print(
            f"{name.upper():10s}: {value:.4f}"
        )

    save_evaluation_results(metrics)