"""
qsofa_rules.py

Calculates qSOFA score for a patient.
"""

def evaluate_qsofa(respiratory_rate, systolic_bp, mental_status):
    """
    Calculate qSOFA score based on vitals.
    Returns score and list of triggered rules.
    """
    score = 0
    triggers = []

    if respiratory_rate >= 22:
        score += 1
        triggers.append("Respiratory rate ≥ 22")

    if systolic_bp <= 100:
        score += 1
        triggers.append("Systolic BP ≤ 100")

    if mental_status != "Alert":
        score += 1
        triggers.append("Altered mental status")

    return score, triggers

# Temporary test
if __name__ == "__main__":
    s, t = evaluate_qsofa(24, 95, "Confused")
    print("Score:", s)
    print("Triggers:", t)
