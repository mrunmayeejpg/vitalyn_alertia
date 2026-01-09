# logic/feature_extraction.py
"""
ML Feature Extraction Module
Converts patient vital history into features for AI risk prediction.
"""

def extract_features(vital_history):
    """
    Converts patient vital history into ML features.

    Parameters:
    -----------
    vital_history : list of dict
        Each dict contains patient vitals at a time point:
        - "rr": respiratory rate
        - "sbp": systolic BP
        - "gcs": Glasgow Coma Scale / mental status
        - "qsofa": qSOFA score

    Returns:
    --------
    features : list of float
        [latest_rr, latest_sbp, latest_gcs, avg_rr, avg_sbp, avg_qsofa]
    """
    if not vital_history:
        return [0, 0, 0, 0, 0, 0]  # fallback if no data

    latest = vital_history[-1]

    avg_rr = sum(v["rr"] for v in vital_history) / len(vital_history)
    avg_sbp = sum(v["sbp"] for v in vital_history) / len(vital_history)
    avg_qsofa = sum(v["qsofa"] for v in vital_history) / len(vital_history)

    return [
        latest["rr"],
        latest["sbp"],
        latest["gcs"],
        avg_rr,
        avg_sbp,
        avg_qsofa
    ]

# ------------------ TEST ------------------
if __name__ == "__main__":
    sample_history = [
        {"rr": 20, "sbp": 110, "gcs": 15, "qsofa": 1},
        {"rr": 22, "sbp": 105, "gcs": 15, "qsofa": 2},
        {"rr": 24, "sbp": 100, "gcs": 14, "qsofa": 2},
    ]
    features = extract_features(sample_history)
    print("Extracted features:", features)
