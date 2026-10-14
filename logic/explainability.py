"""
Explainability logic for Vitalyn Alertia.

Generates human-readable explanations from physiological
measurements and their recent trends.

No qSOFA.
No mental-status feature.
No diagnosis or treatment recommendations.
"""


def generate_explanation(
    current_values,
    trends=None,
    risk_score=None,
):
    """
    Generate a simple explanation of the current AI risk score.

    Parameters
    ----------
    current_values : dict
        Current physiological values. Expected keys:
        resp, bp, hr, temp

    trends : dict, optional
        Recent changes in physiological values. Expected keys:
        resp_trend, bp_trend, hr_trend, temp_trend

    risk_score : float, optional
        Model-predicted risk between 0 and 1.

    Returns
    -------
    list[str]
        Human-readable explanation messages.
    """

    if trends is None:
        trends = {}

    explanation = []

    if risk_score is not None:
        risk_score = float(risk_score)

        if risk_score >= 0.75:
            explanation.append(
                f"AI risk score is high ({risk_score:.2f})."
            )
        elif risk_score >= 0.45:
            explanation.append(
                f"AI risk score is moderate ({risk_score:.2f})."
            )
        else:
            explanation.append(
                f"AI risk score is currently low ({risk_score:.2f})."
            )

    # Respiratory rate
    resp = current_values.get("resp")
    if resp is not None:
        if resp >= 22:
            explanation.append(
                f"Respiratory rate is elevated ({resp:.1f} breaths/min)."
            )
        elif resp < 12:
            explanation.append(
                f"Respiratory rate is low ({resp:.1f} breaths/min)."
            )

    # Systolic blood pressure
    bp = current_values.get("bp")
    if bp is not None:
        if bp <= 100:
            explanation.append(
                f"Systolic blood pressure is low ({bp:.1f} mmHg)."
            )
        elif bp >= 160:
            explanation.append(
                f"Systolic blood pressure is elevated ({bp:.1f} mmHg)."
            )

    # Heart rate
    hr = current_values.get("hr")
    if hr is not None:
        if hr >= 100:
            explanation.append(
                f"Heart rate is elevated ({hr:.1f} bpm)."
            )
        elif hr < 60:
            explanation.append(
                f"Heart rate is low ({hr:.1f} bpm)."
            )

    # Temperature
    temp = current_values.get("temp")
    if temp is not None:
        if temp >= 38.0:
            explanation.append(
                f"Temperature is elevated ({temp:.1f} °C)."
            )
        elif temp < 36.0:
            explanation.append(
                f"Temperature is low ({temp:.1f} °C)."
            )

    # Trends
    resp_trend = trends.get("resp_trend")
    if resp_trend is not None and resp_trend > 2:
        explanation.append(
            f"Respiratory rate has increased by {resp_trend:.1f}."
        )

    bp_trend = trends.get("bp_trend")
    if bp_trend is not None and bp_trend < -10:
        explanation.append(
            f"Systolic blood pressure has decreased by "
            f"{abs(bp_trend):.1f} mmHg."
        )

    hr_trend = trends.get("hr_trend")
    if hr_trend is not None and hr_trend > 10:
        explanation.append(
            f"Heart rate has increased by {hr_trend:.1f} bpm."
        )

    temp_trend = trends.get("temp_trend")
    if temp_trend is not None and temp_trend > 0.5:
        explanation.append(
            f"Temperature has increased by {temp_trend:.1f} °C."
        )

    if not explanation:
        explanation.append(
            "No major physiological abnormalities were identified "
            "from the available measurements."
        )

    explanation.append(
        "⚠️ This system provides AI-based risk estimation and "
        "does not constitute a medical diagnosis or treatment recommendation."
    )

    return explanation