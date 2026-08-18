import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Vitalyn Alertia",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from logic.ai_predict import predict_escalation


# ============================================================
# AUTO REFRESH
# ============================================================

try:
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(
        interval=4000,
        key="vitalyn_refresh"
    )

except ImportError:
    pass


# ============================================================
# HEADER
# ============================================================

st.title("🩺 Vitalyn Alertia")

st.caption(
    "AI-Based Early Patient Risk Monitoring System"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("👨‍⚕️ Doctor Panel")

    doctors = [
        "Dr. A. Mehta",
        "Dr. S. Rao",
        "Dr. K. Sharma",
        "Dr. R. Iyer",
        "Dr. P. Kulkarni",
        "Dr. N. Verma",
        "Dr. M. Das",
        "Dr. T. Sen",
        "Dr. H. Kapoor",
        "Dr. J. Malhotra"
    ]

    selected_doctor = st.selectbox(
        "Doctor on Duty",
        doctors
    )

    st.markdown(
        f"**Currently Viewing:** {selected_doctor}"
    )

    st.divider()

    st.info(
        "This system is an academic prototype for "
        "early risk monitoring and is not a medical diagnosis."
    )


# ============================================================
# GENERATE PATIENTS
# ============================================================

def generate_patients(n=60):

    patients = []

    for i in range(n):

        patients.append({

            "patient_id": f"P-{i + 1:03}",

            "age": np.random.randint(
                18,
                90
            ),

            "rr": np.random.randint(
                16,
                26
            ),

            "bp": np.random.randint(
                90,
                140
            ),

            "hr": np.random.randint(
                60,
                120
            ),

            "temp": round(
                np.random.uniform(
                    36.5,
                    38.5
                ),
                1
            ),

            "mental": "Alert"
        })

    return pd.DataFrame(patients)


# ============================================================
# SESSION STATE
# ============================================================

# ------------------------------------------------------------
# Patient data
# ------------------------------------------------------------

if "patients" not in st.session_state:

    st.session_state.patients = generate_patients()

else:

    # Fix old session data created before age was added
    if "age" not in st.session_state.patients.columns:

        st.session_state.patients["age"] = np.random.randint(
            18,
            90,
            len(st.session_state.patients)
        )


# ------------------------------------------------------------
# History
# ------------------------------------------------------------

if "history" not in st.session_state:

    st.session_state.history = {}


# Make sure every patient has history

for pid in st.session_state.patients["patient_id"]:

    if pid not in st.session_state.history:

        st.session_state.history[pid] = []


# ------------------------------------------------------------
# Selected patient
# ------------------------------------------------------------

if "selected_patient" not in st.session_state:

    st.session_state.selected_patient = None


# ============================================================
# CURRENT PATIENT DATA
# ============================================================

df = st.session_state.patients.copy()


# Safety check for age

if "age" not in df.columns:

    df["age"] = np.random.randint(
        18,
        90,
        len(df)
    )


# ============================================================
# LIVE VITAL SIMULATION
# ============================================================

for i in df.index:

    # Respiratory rate
    df.at[i, "rr"] = max(
        8,
        int(
            df.at[i, "rr"]
            + np.random.randint(-1, 2)
        )
    )

    # Blood pressure
    df.at[i, "bp"] = max(
        60,
        int(
            df.at[i, "bp"]
            + np.random.randint(-3, 4)
        )
    )

    # Heart rate
    df.at[i, "hr"] = max(
        40,
        int(
            df.at[i, "hr"]
            + np.random.randint(-2, 3)
        )
    )

    # Temperature
    df.at[i, "temp"] = round(
        np.clip(
            df.at[i, "temp"]
            + np.random.uniform(
                -0.05,
                0.05
            ),
            35,
            41
        ),
        1
    )

    # Mental status
    if (
        df.at[i, "rr"] >= 22
        or df.at[i, "bp"] <= 100
    ):

        df.at[i, "mental"] = "Confused"

    else:

        df.at[i, "mental"] = "Alert"


# ============================================================
# QSOFA CALCULATION
# ============================================================

qsofa_scores = []

for _, patient in df.iterrows():

    score = (

        int(
            patient["rr"] >= 22
        )

        +

        int(
            patient["bp"] <= 100
        )

        +

        int(
            patient["mental"] == "Confused"
        )
    )

    qsofa_scores.append(score)


df["qSOFA"] = qsofa_scores


# ============================================================
# ML RISK PREDICTION
# ============================================================

risk_scores = []

for _, patient in df.iterrows():

    # ========================================================
    # EXACT FEATURES USED DURING MODEL TRAINING
    #
    # 1. resp
    # 2. bp
    # 3. hr
    # 4. temp
    # 5. age
    # 6. qsofa
    # ========================================================

    features = {

        "resp": float(
            patient["rr"]
        ),

        "bp": float(
            patient["bp"]
        ),

        "hr": float(
            patient["hr"]
        ),

        "temp": float(
            patient["temp"]
        ),

        "age": float(
            patient["age"]
        ),

        "qsofa": float(
            patient["qSOFA"]
        )
    }


    try:

        risk = predict_escalation(
            features
        )

        risk = float(
            np.clip(
                risk,
                0,
                1
            )
        )

    except Exception as error:

        st.error(
            f"ML prediction error: {error}"
        )

        risk = 0.0


    risk_scores.append(
        risk
    )


df["AI_Risk"] = risk_scores


# ============================================================
# STATUS
# ============================================================

def get_status(risk):

    if risk >= 0.75:

        return "Critical"

    elif risk >= 0.45:

        return "Moderate"

    else:

        return "Stable"


df["Status"] = [

    get_status(
        risk
    )

    for risk in df["AI_Risk"]
]


# ============================================================
# UPDATE HISTORY
# ============================================================

for _, patient in df.iterrows():

    pid = patient["patient_id"]

    if pid not in st.session_state.history:

        st.session_state.history[pid] = []


    st.session_state.history[pid].append({

        "rr":
            patient["rr"],

        "bp":
            patient["bp"],

        "hr":
            patient["hr"],

        "temp":
            patient["temp"],

        "risk":
            patient["AI_Risk"]
    })


    # Keep last 30 readings

    st.session_state.history[pid] = (
        st.session_state.history[pid][-30:]
    )


# ============================================================
# SORT PATIENTS BY RISK
# ============================================================

df = df.sort_values(
    "AI_Risk",
    ascending=False
).reset_index(
    drop=True
)


# Save updated patient data

st.session_state.patients = df


# ============================================================
# SUMMARY COUNTS
# ============================================================

critical_count = len(
    df[
        df["Status"] == "Critical"
    ]
)

moderate_count = len(
    df[
        df["Status"] == "Moderate"
    ]
)

stable_count = len(
    df[
        df["Status"] == "Stable"
    ]
)


# ============================================================
# SUMMARY
# ============================================================

st.subheader(
    "📊 Live Monitoring Summary"
)

summary1, summary2, summary3, summary4 = st.columns(4)


with summary1:

    st.metric(
        "Total Patients",
        len(df)
    )


with summary2:

    st.metric(
        "🔴 Critical",
        critical_count
    )


with summary3:

    st.metric(
        "🟠 Moderate",
        moderate_count
    )


with summary4:

    st.metric(
        "🟢 Stable",
        stable_count
    )


st.divider()


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns(
    [3, 1]
)


# ============================================================
# PATIENT BOARD
# ============================================================

with left:

    st.subheader(
        "🚨 Live Patient Board"
    )

    st.caption(
        "Patients are ranked according to predicted ML risk."
    )


    patient_columns = st.columns(3)


    for index, patient in df.iterrows():

        pid = patient["patient_id"]

        status = patient["Status"]

        risk = patient["AI_Risk"]


        with patient_columns[
            index % 3
        ]:

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            if status == "Critical":

                st.error(
                    "🔴 CRITICAL"
                )

            elif status == "Moderate":

                st.warning(
                    "🟠 MODERATE"
                )

            else:

                st.success(
                    "🟢 STABLE"
                )


            # ------------------------------------------------
            # Patient ID
            # ------------------------------------------------

            st.markdown(
                f"### {pid}"
            )


            st.write(
                f"**Age:** {patient['age']} years"
            )


            # ------------------------------------------------
            # Vitals
            # ------------------------------------------------

            vital_col1, vital_col2 = st.columns(2)


            with vital_col1:

                st.write(
                    f"**RR:** {patient['rr']}"
                )

                st.write(
                    f"**HR:** {patient['hr']}"
                )


            with vital_col2:

                st.write(
                    f"**BP:** {patient['bp']}"
                )

                st.write(
                    f"**Temp:** {patient['temp']} °C"
                )


            st.write(
                f"**qSOFA:** {patient['qSOFA']}"
            )


            # ------------------------------------------------
            # Risk
            # ------------------------------------------------

            st.write(
                f"**AI Risk: {risk * 100:.1f}%**"
            )


            st.progress(
                float(risk)
            )


            # ------------------------------------------------
            # Details button
            # ------------------------------------------------

            if st.button(
                "View Details",
                key=f"view_{pid}"
            ):

                st.session_state.selected_patient = pid

                st.rerun()


            st.divider()


# ============================================================
# ANALYTICS PANEL
# ============================================================

with right:

    st.subheader(
        "📈 Patient Analytics"
    )


    selected_pid = (
        st.session_state.selected_patient
    )


    if selected_pid is None:

        st.info(
            "Select a patient from the "
            "patient board to view detailed analytics."
        )


    else:

        selected_rows = df[
            df["patient_id"]
            == selected_pid
        ]


        if selected_rows.empty:

            st.warning(
                "Patient not found."
            )

        else:

            patient = selected_rows.iloc[0]


            # ------------------------------------------------
            # Patient
            # ------------------------------------------------

            st.markdown(
                f"## {patient['patient_id']}"
            )


            st.write(
                f"**Doctor:** {selected_doctor}"
            )


            st.write(
                f"**Age:** {patient['age']} years"
            )


            st.write(
                f"**Status:** {patient['Status']}"
            )


            st.write(
                f"**qSOFA:** {patient['qSOFA']}"
            )


            # ------------------------------------------------
            # Risk
            # ------------------------------------------------

            risk = patient["AI_Risk"]


            st.metric(
                "AI Risk Probability",
                f"{risk * 100:.1f}%"
            )


            st.progress(
                float(risk)
            )


            if risk >= 0.75:

                st.error(
                    "⚠️ High predicted risk. "
                    "Patient requires immediate attention."
                )

            elif risk >= 0.45:

                st.warning(
                    "⚠️ Moderate predicted risk. "
                    "Patient should be monitored closely."
                )

            else:

                st.success(
                    "✓ Lower predicted risk. "
                    "Continue routine monitoring."
                )


            st.divider()


            # ------------------------------------------------
            # Current Vitals
            # ------------------------------------------------

            st.markdown(
                "### Current Vital Signs"
            )


            vital_col1, vital_col2 = st.columns(2)


            with vital_col1:

                st.metric(
                    "Resp. Rate",
                    patient["rr"]
                )

                st.metric(
                    "Heart Rate",
                    patient["hr"]
                )


            with vital_col2:

                st.metric(
                    "Systolic BP",
                    patient["bp"]
                )

                st.metric(
                    "Temperature",
                    f"{patient['temp']} °C"
                )


            st.divider()


            # ------------------------------------------------
            # Historical Trends
            # ------------------------------------------------

            st.markdown(
                "### 📈 Vital Trends"
            )


            history = (
                st.session_state.history[
                    selected_pid
                ]
            )


            if len(history) > 1:

                hist = pd.DataFrame(
                    history
                )


                # --------------------------------------------
                # Respiratory Rate
                # --------------------------------------------

                fig1, ax1 = plt.subplots(
                    figsize=(6, 3)
                )


                ax1.plot(
                    hist["rr"],
                    linewidth=2,
                    label="Respiratory Rate"
                )


                ax1.set_xlabel(
                    "Reading"
                )

                ax1.set_ylabel(
                    "RR"
                )

                ax1.set_title(
                    "Respiratory Rate Trend"
                )


                ax1.grid(
                    True,
                    linestyle="--",
                    alpha=0.4
                )


                ax1.legend()


                st.pyplot(
                    fig1,
                    use_container_width=True
                )


                plt.close(fig1)


                # --------------------------------------------
                # Blood Pressure
                # --------------------------------------------

                fig2, ax2 = plt.subplots(
                    figsize=(6, 3)
                )


                ax2.plot(
                    hist["bp"],
                    linewidth=2,
                    label="Systolic BP"
                )


                ax2.set_xlabel(
                    "Reading"
                )

                ax2.set_ylabel(
                    "BP"
                )

                ax2.set_title(
                    "Blood Pressure Trend"
                )


                ax2.grid(
                    True,
                    linestyle="--",
                    alpha=0.4
                )


                ax2.legend()


                st.pyplot(
                    fig2,
                    use_container_width=True
                )


                plt.close(fig2)


                # --------------------------------------------
                # AI Risk
                # --------------------------------------------

                fig3, ax3 = plt.subplots(
                    figsize=(6, 3)
                )


                ax3.plot(
                    hist["risk"] * 100,
                    linewidth=2,
                    label="AI Risk"
                )


                ax3.set_xlabel(
                    "Reading"
                )

                ax3.set_ylabel(
                    "Risk (%)"
                )

                ax3.set_title(
                    "AI Risk Trend"
                )


                ax3.set_ylim(
                    0,
                    100
                )


                ax3.grid(
                    True,
                    linestyle="--",
                    alpha=0.4
                )


                ax3.legend()


                st.pyplot(
                    fig3,
                    use_container_width=True
                )


                plt.close(fig3)


            else:

                st.info(
                    "Collecting historical readings..."
                )


            # ------------------------------------------------
            # Clear selection
            # ------------------------------------------------

            if st.button(
                "Clear Selection",
                key="clear_patient"
            ):

                st.session_state.selected_patient = None

                st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Vitalyn Alertia | Machine Learning-based "
    "Early Patient Risk Monitoring | Academic Prototype"
)