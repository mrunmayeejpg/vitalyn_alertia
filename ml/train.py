"""
Train Vitalyn Alertia using the PhysioNet 2019 Sepsis Dataset.

- Uses real ICU patient data
- Uses SepsisLabel as target
- No qSOFA
- No mental status
- Patient-level train/test split
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from ml.features import extract_features, FEATURE_NAMES


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

DATA_DIR = os.path.join(
    "data",
    "raw",
    "training_setA"
)

MODEL_DIR = os.path.join(
    "ml",
    "saved_models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sepsis_model.pkl"
)

COMPARISON_PATH = os.path.join(
    "ml",
    "model_comparison.csv"
)


# ============================================================
# GET PATIENT FILES
# ============================================================

def get_patient_files():

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    files = sorted(
        glob.glob(
            os.path.join(DATA_DIR, "*.psv")
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No .psv files found in {DATA_DIR}"
        )

    print(f"Found {len(files):,} patient files.")

    return files


# ============================================================
# BUILD SAMPLES FOR ONE PATIENT
# ============================================================

def build_patient_samples(file_path):

    try:
        df = pd.read_csv(
            file_path,
            sep="|"
        )

    except Exception as error:
        print(f"Could not read {file_path}: {error}")
        return [], []

    required_columns = [
        "Resp",
        "SBP",
        "HR",
        "Temp",
        "Age",
        "SepsisLabel"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        print(
            f"Skipping {file_path}. "
            f"Missing columns: {missing}"
        )
        return [], []

    features = []
    labels = []

    for i in range(len(df)):

        history = df.iloc[:i + 1].copy()

        try:
            feature_vector = extract_features(history)

        except ValueError:
            continue

        label = df.iloc[i]["SepsisLabel"]

        if pd.isna(label):
            continue

        features.append(feature_vector)
        labels.append(int(label))

    return features, labels


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(patient_files):

    all_features = []
    all_labels = []

    total_patients = len(patient_files)

    for index, file_path in enumerate(
        patient_files,
        start=1
    ):

        if index == 1 or index % 100 == 0:
            print(
                f"Processing patient "
                f"{index:,}/{total_patients:,}"
            )

        features, labels = build_patient_samples(
            file_path
        )

        all_features.extend(features)
        all_labels.extend(labels)

    X = pd.DataFrame(
        all_features,
        columns=FEATURE_NAMES
    )

    y = pd.Series(
        all_labels,
        name="SepsisLabel"
    )

    return X, y


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(X_train, y_train):

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }

    trained_models = {}

    for name, model in models.items():

        print("\n" + "=" * 60)
        print(f"Training {name}")
        print("=" * 60)

        if name == "Gradient Boosting":

            class_counts = (
                y_train
                .value_counts()
                .to_dict()
            )

            weight_0 = (
                len(y_train)
                /
                (2 * class_counts.get(0, 1))
            )

            weight_1 = (
                len(y_train)
                /
                (2 * class_counts.get(1, 1))
            )

            sample_weights = y_train.map({
                0: weight_0,
                1: weight_1
            })

            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights
            )

        else:

            model.fit(
                X_train,
                y_train
            )

        trained_models[name] = model

    return trained_models


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    print(f"\nModel: {name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("VITALYN ALERTIA - REAL DATA TRAINING")
    print("=" * 70)

    # Get all patient files
    patient_files = get_patient_files()

    # IMPORTANT:
    # Split patients BEFORE generating hourly samples.
    # This prevents the same patient appearing in both
    # training and testing sets.
    train_files, test_files = train_test_split(
        patient_files,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    print(
        f"\nTraining patients: {len(train_files):,}"
    )

    print(
        f"Testing patients : {len(test_files):,}"
    )

    # --------------------------------------------------------
    # BUILD TRAINING DATA
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING TRAINING DATA")
    print("=" * 70)

    X_train, y_train = build_dataset(
        train_files
    )

    print(
        f"\nTraining samples: {len(X_train):,}"
    )

    print(
        f"Training features: {len(FEATURE_NAMES)}"
    )

    print("\nTraining class distribution:")

    print(
        y_train.value_counts()
    )

    # --------------------------------------------------------
    # BUILD TEST DATA
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING TEST DATA")
    print("=" * 70)

    X_test, y_test = build_dataset(
        test_files
    )

    print(
        f"\nTesting samples: {len(X_test):,}"
    )

    print("\nTesting class distribution:")

    print(
        y_test.value_counts()
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if X_train.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if X_test.empty:
        raise ValueError(
            "Testing dataset is empty."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "Training dataset contains only one class."
        )

    if y_test.nunique() < 2:
        raise ValueError(
            "Testing dataset contains only one class."
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    trained_models = train_models(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    results = []

    for name, model in trained_models.items():

        result = evaluate_model(
            name,
            model,
            X_test,
            y_test
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        "F1",
        ascending=False
    ).reset_index(
        drop=True
    )

    best_model_name = (
        results_df.iloc[0]["Model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # SAVE MODEL COMPARISON
    # --------------------------------------------------------

    results_df.to_csv(
        COMPARISON_PATH,
        index=False
    )

    print(
        f"Model comparison saved to: "
        f"{COMPARISON_PATH}"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    print("\nFEATURES USED:")

    for index, feature in enumerate(
        FEATURE_NAMES,
        start=1
    ):
        print(
            f"{index}. {feature}"
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()