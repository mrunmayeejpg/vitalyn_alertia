def calculate_qsofa(vital):
    """
    Calculate qSOFA score.
    """

    score = 0

    if vital["rr"] >= 22:
        score += 1

    if vital["bp"] <= 100:
        score += 1

    if vital["mental"].lower() != "alert":
        score += 1

    return score


def extract_features(vital_history):
    """
    Convert patient vital history into ML features.
    """

    if not vital_history:
        raise ValueError("Vital history cannot be empty.")

    qsofa_history = [
        calculate_qsofa(vital)
        for vital in vital_history
    ]

    latest = vital_history[-1]

    latest_rr = latest["rr"]
    latest_bp = latest["bp"]
    latest_hr = latest["hr"]
    latest_temp = latest["temp"]

    mental_status = (
        0 if latest["mental"].lower() == "alert"
        else 1
    )

    avg_rr = sum(v["rr"] for v in vital_history) / len(vital_history)
    avg_bp = sum(v["bp"] for v in vital_history) / len(vital_history)
    avg_hr = sum(v["hr"] for v in vital_history) / len(vital_history)
    avg_temp = sum(v["temp"] for v in vital_history) / len(vital_history)
    avg_qsofa = sum(qsofa_history) / len(qsofa_history)

    if len(vital_history) >= 2:
        previous = vital_history[-2]

        rr_trend = latest["rr"] - previous["rr"]
        bp_trend = latest["bp"] - previous["bp"]
        hr_trend = latest["hr"] - previous["hr"]
        temp_trend = latest["temp"] - previous["temp"]
    else:
        rr_trend = 0
        bp_trend = 0
        hr_trend = 0
        temp_trend = 0

    return [
        latest_rr,
        latest_bp,
        latest_hr,
        latest_temp,
        mental_status,
        avg_rr,
        avg_bp,
        avg_hr,
        avg_temp,
        avg_qsofa,
        rr_trend,
        bp_trend,
        hr_trend,
        temp_trend
    ]