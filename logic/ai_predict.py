"""
AI prediction logic for Vitalyn Alertia.

No qSOFA.
No mental-status feature.
"""

import os
import joblib
import numpy as np
import pandas as pd

from ml.features import FEATURE_NAMES


MODEL_PATH = os.path.join(
    "ml",
    "saved_models",
    "sepsis_model.pkl"
)


# ============================================================
# LOAD MODEL ONCE
# ============================================================

_MODEL = None


def load_model():

    global _MODEL

    if _MODEL is None:

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                f"Run python ml/train.py first."
            )

        _MODEL = joblib.load(
            MODEL_PATH
        )

    return _MODEL


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(features):

    if isinstance(features, dict):

        missing = [
            name
            for name in FEATURE_NAMES
            if name not in features
        ]

        if missing:
            raise ValueError(
                "Missing required features: "
                + ", ".join(missing)
            )

        values = [
            features[name]
            for name in FEATURE_NAMES
        ]

    elif isinstance(
        features,
        (list, tuple, np.ndarray)
    ):

        if len(features) != len(FEATURE_NAMES):

            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} "
                f"features, received {len(features)}."
            )

        values = list(features)

    else:

        raise TypeError(
            "Features must be a dictionary, "
            "list, tuple or numpy array."
        )

    numeric_values = []

    for name, value in zip(
        FEATURE_NAMES,
        values
    ):

        try:
            value = float(value)

        except (TypeError, ValueError):

            raise ValueError(
                f"Feature '{name}' must be numeric."
            )

        if not np.isfinite(value):

            raise ValueError(
                f"Feature '{name}' contains "
                f"an invalid value."
            )

        numeric_values.append(value)

    return pd.DataFrame(
        [numeric_values],
        columns=FEATURE_NAMES
    )


# ============================================================
# PREDICT RISK
# ============================================================

def predict_escalation(features):

    model = load_model()

    X = validate_features(
        features
    )

    probability = model.predict_proba(
        X
    )[0][1]

    return float(
        np.clip(
            probability,
            0.0,
            1.0
        )
    )


# ============================================================
# PREDICT CLASS
# ============================================================

def predict_class(features):

    model = load_model()

    X = validate_features(
        features
    )

    prediction = model.predict(
        X
    )[0]

    return int(prediction)
