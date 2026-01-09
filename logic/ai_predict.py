import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# ------------------ DATA GENERATION ------------------
def generate_training_data(n=10000):
    np.random.seed(42)

    data = {
        "resp": np.random.normal(22, 4, n),
        "bp": np.random.normal(105, 15, n),
        "hr": np.random.normal(95, 18, n),
        "temp": np.random.normal(37.2, 0.7, n),
        "age": np.random.randint(18, 90, n),
        "qsofa": np.random.randint(0, 4, n),
    }

    df = pd.DataFrame(data)

    # Trend-based risk label (NOT diagnosis)
    df["risk"] = (
        (df["resp"] > 24).astype(int) +
        (df["bp"] < 95).astype(int) +
        (df["hr"] > 110).astype(int) +
        (df["temp"] > 38).astype(int) +
        (df["qsofa"] >= 2).astype(int)
    ) >= 2

    return df

# ------------------ TRAIN MODEL ------------------
df = generate_training_data()
X = df[["resp", "bp", "hr", "temp", "age", "qsofa"]]
y = df["risk"]

model = GradientBoostingClassifier()
model.fit(X, y)

# ------------------ PREDICTION ------------------
def predict_escalation(resp, bp, hr, temp, age, qsofa):
    X_new = pd.DataFrame([{
        "resp": resp,
        "bp": bp,
        "hr": hr,
        "temp": temp,
        "age": age,
        "qsofa": qsofa
    }])
    return model.predict_proba(X_new)[0][1]
