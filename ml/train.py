import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

FEATURE_NAMES = [
    "resp",
    "bp",
    "hr",
    "temp",
    "age",
    "qsofa"
]


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def generate_training_data(n=10000):

    np.random.seed(RANDOM_STATE)

    data = {
        "resp": np.random.normal(21, 4.5, n),
        "bp": np.random.normal(110, 16, n),
        "hr": np.random.normal(92, 18, n),
        "temp": np.random.normal(37.1, 0.65, n),
        "age": np.random.randint(18, 90, n),
        "qsofa": np.random.randint(0, 4, n)
    }

    df = pd.DataFrame(data)

    # ========================================================
    # REALISTIC VALUE RANGES
    # ========================================================

    df["resp"] = df["resp"].clip(8, 45)

    df["bp"] = df["bp"].clip(60, 180)

    df["hr"] = df["hr"].clip(40, 180)

    df["temp"] = df["temp"].clip(35, 41)


    # ========================================================
    # SYNTHETIC LATENT RISK
    # ========================================================
    #
    # This is NOT a clinical diagnosis.
    #
    # These labels are temporary and are only being used to
    # test the ML pipeline until a real dataset is integrated.
    # ========================================================

    score = (

        0.45 * (
            (df["resp"] - 20) / 5
        )

        - 0.40 * (
            (df["bp"] - 110) / 20
        )

        + 0.25 * (
            (df["hr"] - 90) / 20
        )

        + 0.35 * (
            (df["temp"] - 37) / 0.7
        )

        + 0.20 * (
            (df["age"] - 50) / 30
        )

        + 0.80 * df["qsofa"]
    )


    # ========================================================
    # ADD RANDOM VARIATION
    # ========================================================

    score += np.random.normal(
        0,
        0.7,
        n
    )


    # ========================================================
    # CONVERT SCORE TO PROBABILITY
    # ========================================================

    probability = 1 / (
        1 + np.exp(-score)
    )


    # ========================================================
    # CREATE BINARY TARGET
    #
    # 0 = Lower risk
    # 1 = Higher risk
    # ========================================================

    df["risk"] = (
        np.random.random(n) < probability
    ).astype(int)


    return df


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models():

    print("=" * 60)
    print("VITALYN ALERTIA - ML TRAINING")
    print("=" * 60)


    # ========================================================
    # GENERATE DATA
    # ========================================================

    df = generate_training_data()


    print(
        f"\nDataset size: {len(df)}"
    )


    print(
        f"Number of features: {len(FEATURE_NAMES)}"
    )


    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print(
        "\nClass distribution:"
    )

    print(
        df["risk"].value_counts()
    )


    # ========================================================
    # FEATURES AND TARGET
    # ========================================================

    X = df[FEATURE_NAMES]

    y = df["risk"]


    # ========================================================
    # TRAIN-TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=y
    )


    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )


    # ========================================================
    # DEFINE ML MODELS
    # ========================================================

    models = {

        "Logistic Regression":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE
                )
            )
        ]),


        "Random Forest":

        RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),


        "Gradient Boosting":

        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }


    # ========================================================
    # TRAIN AND EVALUATE
    # ========================================================

    results = []

    trained_models = {}


    for name, model in models.items():

        print(
            f"\n{name}"
        )

        print(
            "-" * len(name)
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )


        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        results.append({

            "Model": name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1": f1,

            "ROC-AUC": roc_auc
        })


        trained_models[name] = model


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    print(
        "\n" + "=" * 60
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 60
    )


    print(
        results_df.to_string(
            index=False
        )
    )


    # ========================================================
    # SELECT BEST MODEL
    # ========================================================
    #
    # F1 is used for model selection because this is a
    # risk-screening problem where both precision and recall
    # are important.
    # ========================================================

    best_model_name = (

        results_df
        .sort_values(
            "F1",
            ascending=False
        )
        .iloc[0]["Model"]
    )


    best_model = trained_models[
        best_model_name
    ]


    print(
        "\n" + "=" * 60
    )

    print(
        f"BEST MODEL: {best_model_name}"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    model_directory = os.path.join(
        os.path.dirname(__file__),
        "saved_models"
    )


    os.makedirs(
        model_directory,
        exist_ok=True
    )


    model_path = os.path.join(
        model_directory,
        "sepsis_model.pkl"
    )


    joblib.dump(
        best_model,
        model_path
    )


    print(
        "\nModel saved to:"
    )

    print(
        model_path
    )


    # ========================================================
    # SAVE MODEL COMPARISON
    # ========================================================

    results_path = os.path.join(
        os.path.dirname(__file__),
        "model_comparison.csv"
    )


    results_df.to_csv(
        results_path,
        index=False
    )


    print(
        "\nModel comparison saved to:"
    )

    print(
        results_path
    )


    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_models()