"""
test_integration.py

End-to-end integration test:
- Loads synthetic patient data
- Applies qSOFA rule-based logic at each time point
- Detects escalation trends
- Prints explainable outputs
- Visualizes qSOFA timeline

This file contains NO UI code.
"""

from utils.data_loader import load_patient
from logic.qsofa_adapter import calculate_qsofa
# Adapt your original function name
from logic.risk_escalation import assess_risk_escalation as detect_escalation
from visualization.timeline import plot_qsofa_timeline
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load a demo patient scenario
# --------------------------------------------------
df = load_patient("Deteriorating Patient")

# --------------------------------------------------
# Apply qSOFA logic over time
# --------------------------------------------------
qsofa_scores = []
alerts = []

for _, row in df.iterrows():
    vitals = row.to_dict()

    # Calculate qSOFA score and rule triggers
    score, triggers = calculate_qsofa(vitals)
    qsofa_scores.append(score)

    # Detect escalation trend using your original function
    escalation, trend_reasons = detect_escalation(qsofa_scores)

    # Store explainable alert
    alerts.append(
        {
            "time": vitals["time"],
            "qSOFA_score": score,
            "rule_triggers": triggers,
            "escalation": escalation,
            "trend_reasons": trend_reasons,
        }
    )

# --------------------------------------------------
# Attach qSOFA score back to DataFrame
# --------------------------------------------------
df["qSOFA"] = qsofa_scores

# --------------------------------------------------
# Console output (for verification)
# --------------------------------------------------
print("\n=== FINAL DATAFRAME ===")
print(df)

print("\n=== GENERATED ALERTS ===")
for alert in alerts:
    print("\nTime:", alert["time"])
    print("qSOFA Score:", alert["qSOFA_score"])
    print("Triggers:")
    for t in alert["rule_triggers"]:
        print(" -", t)
    print("Escalation:", alert["escalation"])
    print("Trend Reasons:")
    for tr in alert["trend_reasons"]:
        print(" -", tr)

# --------------------------------------------------
# Visualization: qSOFA escalation timeline
# --------------------------------------------------
fig = plot_qsofa_timeline(df)
plt.show()
