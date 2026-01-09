import matplotlib.pyplot as plt


def plot_vital_trends(df):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df["time"], df["resp_rate"], marker="o", label="Respiratory Rate")
    ax.plot(df["time"], df["systolic_bp"], marker="o", label="Systolic BP")

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Vital Sign Trends Over Time")
    ax.legend()
    ax.grid(True)

    return fig
