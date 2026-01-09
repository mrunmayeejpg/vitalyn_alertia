from logic.qsofa_rules import evaluate_qsofa


def calculate_qsofa(vitals_dict):
    """
    Adapter function to convert vitals dictionary
    into arguments required by evaluate_qsofa().
    """

    score, reasons = evaluate_qsofa(
        respiratory_rate=vitals_dict["resp_rate"],
        systolic_bp=vitals_dict["systolic_bp"],
        mental_status=vitals_dict["mental_status"],
    )

    return score, reasons
