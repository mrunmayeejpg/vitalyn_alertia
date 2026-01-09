"""
risk_escalation.py

Analyzes qSOFA score trends for escalation.
"""

def assess_risk_escalation(qsofa_scores):
    escalation = False
    reasons = []

    if len(qsofa_scores) < 2:
        reasons.append("Insufficient data points for trend analysis")
        return escalation, reasons

    latest = qsofa_scores[-1]
    previous = qsofa_scores[-2]

    if latest >= 2:
        escalation = True
        reasons.append("qSOFA score ≥ 2 at latest time point")

    if latest > previous:
        escalation = True
        reasons.append("qSOFA score increased compared to previous measurement")

    return escalation, reasons

# Temporary test
if __name__ == "__main__":
    scores = [1,1,2,3]
    esc, reasons = assess_risk_escalation(scores)
    print("Escalation:", esc)
    print("Reasons:", reasons)
