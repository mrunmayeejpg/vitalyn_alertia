"""
Visualization utilities for Vitalyn Alertia.

Designed for PhysioNet ICU time-series data.
"""

import matplotlib.pyplot as plt


def plot_vital_trends(df):
    """
    Plot major physiological variables over ICU time.

    Expected columns:
        ICULOS
        Resp
        SBP
        HR
        Temp
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    if "ICULOS" not in df.columns:
        x = range(len(df))
        x_label = "Time Step"
    else:
        x = df["ICULOS"]
        x_label = "ICU Length of Stay (hours)"

    if "Resp" in df.columns:
        ax.plot(
            x,
            df["Resp"],
            marker="o",
            label="Respiratory Rate"
        )

    if "SBP" in df.columns:
        ax.plot(
            x,
            df["SBP"],
            marker="o",
            label="Systolic BP"
        )

    if "HR" in df.columns:
        ax.plot(
            x,
            df["HR"],
            marker="o",
            label="Heart Rate"
        )

    if "Temp" in df.columns:
        ax.plot(
            x,
            df["Temp"],
            marker="o",
            label="Temperature"
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Measurement")
    ax.set_title("Patient Physiological Trends")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    return fig