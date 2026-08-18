import os
import joblib


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "saved_models",
    "sepsis_model.pkl"
)


def load_model():
    """
    Load the trained ML model.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. "
            "Run ml/train.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_risk(features):
    """
    Return ML risk probability.
    """

    model = load_model()

    probability = model.predict_proba([features])[0][1]

    return round(float(probability), 2)


def predict_class(features):
    """
    Return predicted class.
    0 = Lower risk
    1 = Higher risk
    """

    model = load_model()

    prediction = model.predict([features])[0]

    return int(prediction)