def extract_features(vital_history):
    """
    Converts patient vital history into ML features
    """

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
