import os
import joblib
import pandas as pd


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "ml",
    "saved_models",
    "sepsis_model.pkl"
)

MODEL_PATH = os.path.abspath(MODEL_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# FEATURE NAMES
# Must exactly match train.py
# ============================================================

FEATURE_NAMES = [
    "resp",
    "bp",
    "hr",
    "temp",
    "age",
    "qsofa"
]


# ============================================================
# PREDICTION
# ============================================================

def predict_escalation(features):
    """
    Predict patient escalation risk.

    Expected features:
        resp
        bp
        hr
        temp
        age
        qsofa
    """

    # --------------------------------------------------------
    # If a dictionary is provided
    # --------------------------------------------------------

    if isinstance(features, dict):

        data = {
            "resp": features["resp"],
            "bp": features["bp"],
            "hr": features["hr"],
            "temp": features["temp"],
            "age": features["age"],
            "qsofa": features["qsofa"]
        }

    # --------------------------------------------------------
    # If a list/array is provided
    # --------------------------------------------------------

    else:

        if len(features) != 6:
            raise ValueError(
                "Expected exactly 6 features: "
                "resp, bp, hr, temp, age, qsofa"
            )

        data = dict(
            zip(FEATURE_NAMES, features)
        )


    # --------------------------------------------------------
    # Create DataFrame with EXACT feature order
    # --------------------------------------------------------

    X_new = pd.DataFrame(
        [data],
        columns=FEATURE_NAMES
    )


    # --------------------------------------------------------
    # Predict probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        X_new
    )[0][1]


    return float(probability)