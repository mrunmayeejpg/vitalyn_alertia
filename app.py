import streamlit as st

# ---------------------------------------------------
# PAGE CONFIG (MUST BE FIRST)
# ---------------------------------------------------
st.set_page_config(page_title="Vitalyn Alertia", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=4000, key="refresh")

# ---------------------------------------------------
# LOGIN SYSTEM
# ---------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown(
        "<h2 style='text-align:center; color:#064e3b;'>🔐 Vitalyn Alertia Login</h2>",
        unsafe_allow_html=True
    )
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "admin" and p == "admin123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

def logout():
    st.session_state.logged_in = False
    st.rerun()

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("🩺Vitalyn Alertia")
st.caption("Live AI-based Patient Risk Monitoring Dashboard")

# ---------------------------------------------------
# SIDEBAR — DOCTOR DROPDOWN
# ---------------------------------------------------
with st.sidebar:
    st.header("👨‍⚕️ Doctor Panel")
    doctors = [
        "Dr. A. Mehta", "Dr. S. Rao", "Dr. K. Sharma", "Dr. R. Iyer",
        "Dr. P. Kulkarni", "Dr. N. Verma", "Dr. M. Das", "Dr. T. Sen"
    ]
    selected_doctor = st.selectbox("Doctor on duty", doctors)

    if st.button("Logout"):
        logout()

# ---------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------
def generate_patients(n=60):
    return pd.DataFrame([
        {
            "patient_id": f"P-{i+1:03}",
            "rr": np.random.randint(16, 26),
            "bp": np.random.randint(90, 140),
            "hr": np.random.randint(60, 120),
            "temp": round(np.random.uniform(36.5, 38.5), 1),
            "mental": "Alert"
        } for i in range(n)
    ])

# ---------------------------------------------------
# SESSION INIT
# ---------------------------------------------------
if "patients" not in st.session_state:
    st.session_state.patients = generate_patients()

if "history" not in st.session_state:
    st.session_state.history = {
        pid: [] for pid in st.session_state.patients["patient_id"]
    }

if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

if "last_positions" not in st.session_state:
    st.session_state.last_positions = {}

if "animation_count" not in st.session_state:
    st.session_state.animation_count = 0

df = st.session_state.patients.copy()

# ---------------------------------------------------
# LIVE SIMULATION
# ---------------------------------------------------
for i in df.index:
    df.at[i, "rr"] = max(10, df.at[i, "rr"] + np.random.randint(-1, 2))
    df.at[i, "bp"] = max(70, df.at[i, "bp"] + np.random.randint(-3, 3))
    df.at[i, "hr"] = max(40, df.at[i, "hr"] + np.random.randint(-2, 2))
    df.at[i, "mental"] = "Confused" if df.at[i, "rr"] > 22 or df.at[i, "bp"] < 100 else "Alert"

# ---------------------------------------------------
# QSOFA + AI RISK
# ---------------------------------------------------
qsofa, risk = [], []

for _, p in df.iterrows():
    score = int(p["rr"] >= 22) + int(p["bp"] <= 100) + int(p["mental"] == "Confused")
    qsofa.append(score)

    r = (
        0.45 * (p["rr"] / 35) +
        0.35 * (1 - p["bp"] / 160) +
        0.20 * (score / 3)
    )
    r = np.clip(r, 0, 1)
    risk.append(r)

    st.session_state.history[p["patient_id"]].append(
        {"rr": p["rr"], "bp": p["bp"], "risk": r}
    )
    st.session_state.history[p["patient_id"]] = st.session_state.history[p["patient_id"]][-30:]

df["qSOFA"] = qsofa
df["AI_Risk"] = risk

# ---------------------------------------------------
# STATUS & SORT
# ---------------------------------------------------
df = df.sort_values("AI_Risk", ascending=False).reset_index(drop=True)

df["Status"] = [
    "Critical" if i < 5 else
    "Moderate" if r > 0.45 else
    "Stable"
    for i, r in enumerate(df["AI_Risk"])
]

st.session_state.patients = df

# ---------------------------------------------------
# STYLES — BUTTON COLORS FIXED
# ---------------------------------------------------
st.markdown("""
<style>
.stApp { background-color:#e8f5e9; color:black; }

.card-wrapper { transition: transform 0.6s ease-in-out; }

.card {
    height: 200px;
    padding: 14px;
    border-radius: 14px;
    background: white;
    box-shadow: 0 6px 14px rgba(0,0,0,0.1);
    color: black;
}

.critical { border-left: 6px solid #dc2626; }
.moderate { border-left: 6px solid #2563eb; }
.stable { border-left: 6px solid #16a34a; }

/* BUTTON STYLING */
button[kind="primary"] {
    background-color: #065f46 !important;
    color: white !important;
    border-radius: 8px;
    font-weight: 600;
}

button[kind="secondary"] {
    background-color: #1e40af !important;
    color: white !important;
    border-radius: 8px;
    font-weight: 600;
}

button:hover {
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------
left, right = st.columns([3, 1])

with left:
    st.subheader("🚨 Live Patient Board 🚨")
    cols = st.columns(3)

    for idx, p in df.iterrows():
        pid = p["patient_id"]

        prev = st.session_state.last_positions.get(pid, idx)
        delta = prev - idx

        move = 0
        if abs(delta) > 0 and st.session_state.animation_count < 2:
            move = delta * 220
            st.session_state.animation_count += 1

        st.session_state.last_positions[pid] = idx

        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="card-wrapper" style="transform: translateY({move}px);">
                    <div class="card {p['Status'].lower()}">
                        <b>{pid}</b><br>
                        RR: {p['rr']} | BP: {p['bp']}<br>
                        HR: {p['hr']} | Temp: {p['temp']}<br>
                        qSOFA: {p['qSOFA']}<br>
                        Risk: {p['AI_Risk']*100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button("View Details", key=pid, type="primary"):
                st.session_state.selected_patient = pid

# ---------------------------------------------------
# ANALYTICS PANEL
# ---------------------------------------------------
with right:
    st.subheader("📊 Patient Analytics")

    if st.session_state.selected_patient:
        pid = st.session_state.selected_patient
        p = df[df["patient_id"] == pid].iloc[0]
        hist = pd.DataFrame(st.session_state.history[pid])

        st.markdown(f"**Patient:** {pid}")
        st.markdown(f"**Doctor:** {selected_doctor}")
        st.markdown(f"**Status:** {p['Status']}")
        st.markdown(f"**Risk:** {p['AI_Risk']*100:.1f}%")

        if p["AI_Risk"] > 0.75:
            st.markdown("<b style='color:black;'>Patient condition very serious</b>", unsafe_allow_html=True)
        elif p["AI_Risk"] > 0.6:
            st.markdown("<b style='color:black;'>Patient needs immediate attention</b>", unsafe_allow_html=True)
        elif p["AI_Risk"] > 0.45:
            st.markdown("<b style='color:black;'>Patient needs close monitoring</b>", unsafe_allow_html=True)
        else:
            st.markdown("<b style='color:black;'>Patient is stable</b>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist["rr"], label="Respiration Rate", linewidth=2)
        ax.plot(hist["bp"], label="Blood Pressure", linewidth=2)
        ax.plot(hist["risk"], label="AI Risk", linewidth=2)

        ax.set_title("Vitals & AI Risk Trend")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if st.button("Clear Selection", type="secondary"):
            st.session_state.selected_patient = None
