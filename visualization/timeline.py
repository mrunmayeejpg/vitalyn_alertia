import matplotlib.pyplot as plt


def plot_qsofa_timeline(df):
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.step(df["time"], df["qSOFA"], where="post", marker="o")
    ax.set_ylim(-0.2, 3.2)

    ax.set_xlabel("Time")
    ax.set_ylabel("qSOFA Score")
    ax.set_title("qSOFA Risk Escalation Timeline")
    ax.grid(True)

    return fig
