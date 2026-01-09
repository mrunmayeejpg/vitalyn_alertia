"""
explainability.py

Generates human-readable explanations for alerts.
"""

def generate_explanation(score, triggers, escalation_reasons):
    explanation = []
    explanation.append(f"Current qSOFA score is {score}.")

    if triggers:
        explanation.append("Contributing factors:")
        for t in triggers:
            explanation.append(f"- {t}")

    if escalation_reasons:
        explanation.append("Risk escalation detected due to:")
        for r in escalation_reasons:
            explanation.append(f"- {r}")

    explanation.append("⚠️ This alert is based solely on screening logic and does not indicate diagnosis or treatment.")
    return explanation

# Temporary test
if __name__ == "__main__":
    score = 3
    triggers = ["Respiratory rate ≥ 22","Systolic BP ≤ 100","Altered mental status"]
    escalation_reasons = ["qSOFA score ≥ 2","Score increased"]
    expl = generate_explanation(score, triggers, escalation_reasons)
    print("\n".join(expl))
