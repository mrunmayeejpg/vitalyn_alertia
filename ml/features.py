"""
Feature extraction for Vitalyn Alertia.

Uses real PhysioNet ICU time-series data.
No qSOFA or mental-status features are used.
"""

import pandas as pd


# ============================================================
# MODEL FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "resp",
    "bp",
    "hr",
    "temp",
    "age",
    "avg_resp",
    "avg_bp",
    "avg_hr",
    "avg_temp",
    "resp_trend",
    "bp_trend",
    "hr_trend",
    "temp_trend",
]


# ============================================================
# HELPER: CLEAN VITAL SERIES
# ============================================================

def clean_series(history, column):
    """
    Forward-fill and backward-fill missing values
    for a physiological variable.

    If the entire column is missing, raise an error.
    """

    series = pd.to_numeric(
        history[column],
        errors="coerce"
    )

    if series.notna().sum() == 0:
        raise ValueError(
            f"No valid values available for {column}."
        )

    return series.ffill().bfill()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(history):
    """
    Convert a patient's hourly PhysioNet history
    into the feature vector used by the ML model.

    Parameters
    ----------
    history : pandas.DataFrame
        Patient's time-series data.

    Returns
    -------
    list
        13 numerical ML features.
    """

    if history is None or history.empty:
        raise ValueError(
            "Patient history cannot be empty."
        )

    required_columns = [
        "Resp",
        "SBP",
        "HR",
        "Temp",
        "Age",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in history.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Clean physiological variables
    # --------------------------------------------------------

    resp = clean_series(history, "Resp")
    bp = clean_series(history, "SBP")
    hr = clean_series(history, "HR")
    temp = clean_series(history, "Temp")

    # --------------------------------------------------------
    # Latest measurements
    # --------------------------------------------------------

    latest_resp = float(resp.iloc[-1])
    latest_bp = float(bp.iloc[-1])
    latest_hr = float(hr.iloc[-1])
    latest_temp = float(temp.iloc[-1])

    # Age is generally constant for a patient.
    age_series = pd.to_numeric(
        history["Age"],
        errors="coerce"
    ).dropna()

    if age_series.empty:
        raise ValueError(
            "No valid age value available."
        )

    age = float(age_series.iloc[-1])

    # --------------------------------------------------------
    # Historical averages
    # --------------------------------------------------------

    avg_resp = float(resp.mean())
    avg_bp = float(bp.mean())
    avg_hr = float(hr.mean())
    avg_temp = float(temp.mean())

    # --------------------------------------------------------
    # One-step trends
    #
    # Positive value  = variable increased
    # Negative value  = variable decreased
    # --------------------------------------------------------

    if len(history) >= 2:

        resp_trend = float(
            resp.iloc[-1] - resp.iloc[-2]
        )

        bp_trend = float(
            bp.iloc[-1] - bp.iloc[-2]
        )

        hr_trend = float(
            hr.iloc[-1] - hr.iloc[-2]
        )

        temp_trend = float(
            temp.iloc[-1] - temp.iloc[-2]
        )

    else:

        resp_trend = 0.0
        bp_trend = 0.0
        hr_trend = 0.0
        temp_trend = 0.0

    # --------------------------------------------------------
    # Final feature vector
    # --------------------------------------------------------

    features = [
        latest_resp,
        latest_bp,
        latest_hr,
        latest_temp,
        age,
        avg_resp,
        avg_bp,
        avg_hr,
        avg_temp,
        resp_trend,
        bp_trend,
        hr_trend,
        temp_trend,
    ]

    return features