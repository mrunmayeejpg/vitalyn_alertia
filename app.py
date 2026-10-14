import streamlit as st
import pandas as pd
import numpy as np

from logic.ai_predict import predict_escalation
from utils.data_loader import list_patients, load_patient
from visualization.charts import plot_vital_trends


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Vitalyn Alertia",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

CRITICAL_THRESHOLD = 0.75
MODERATE_THRESHOLD = 0.45

# Number of cards actually rendered on screen.
# This is NOT the number of patients in the dataset.

# ---- Hospital & Doctor selector (sidebar) ----
# Placeholder data – replace with real source as needed
hospitals = [
    "City General Hospital",
    "St. Mary's Medical Center",
    "Metro Health Clinic",
    "Sunrise Community Hospital",
]

# Simple mapping of doctors per hospital (example)
doctors_by_hospital = {
    "City General Hospital": ["Dr. Alice Smith", "Dr. Bob Jones"],
    "St. Mary's Medical Center": ["Dr. Carol Lee", "Dr. Dan Patel"],
    "Metro Health Clinic": ["Dr. Eva Green", "Dr. Frank Wu"],
    "Sunrise Community Hospital": ["Dr. Grace Kim", "Dr. Henry Liu"],
}

selected_hospital = st.sidebar.selectbox("Select Hospital", hospitals)
selected_doctor = st.sidebar.selectbox(
    "Select Doctor",
    doctors_by_hospital.get(selected_hospital, []),
)

st.sidebar.markdown("---")
st.sidebar.write(f"**Hospital:** {selected_hospital}")
st.sidebar.write(f"**Doctor:** {selected_doctor}")
# This is NOT the number of patients in the dataset.
DISPLAY_COUNT = 30


# ============================================================
# SIMPLE CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        padding: 0;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.72rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.05rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
        background-color: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 6px 20px rgba(0,0,0,0.18);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):

    try:
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):

        return None


def latest_value(df, column):

    if df is None:
        return None

    if column not in df.columns:
        return None

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return None

    return safe_float(
        values.iloc[-1]
    )


def format_value(
    value,
    unit="",
    decimals=1
):

    if value is None:
        return "—"

    return f"{value:.{decimals}f}{unit}"


def get_risk_background(risk):

    if risk >= CRITICAL_THRESHOLD:
        return "🔴"

    if risk >= MODERATE_THRESHOLD:
        return "🟡"

    return "🟢"


# ============================================================
# LOAD PATIENT LIST
# ============================================================

try:

    patient_ids = list_patients()

except Exception as e:

    st.error(
        f"Could not load patient list: {e}"
    )

    st.stop()


if not patient_ids:

    st.warning(
        "No PhysioNet patient files were found."
    )

    st.stop()


# ============================================================
# CACHED PATIENT LOADING
# ============================================================

@st.cache_data(
    show_spinner=False
)
def get_patient_data(patient_id):

    return load_patient(
        patient_id
    )


# ============================================================
# CACHED PREDICTION
# ============================================================

@st.cache_data(
    show_spinner=False
)
def get_prediction(patient_id):

    try:

        df = get_patient_data(
            patient_id
        )

        if df is None or df.empty:
            return None

        from ml.features import extract_features

        features = extract_features(
            df
        )

        risk = predict_escalation(
            features
        )

        return float(risk)

    except Exception:

        return None


# ============================================================
# SESSION STATE
# ============================================================

if "predictions" not in st.session_state:

    st.session_state.predictions = {}


if "predictions_loaded" not in st.session_state:

    st.session_state.predictions_loaded = False


# ============================================================
# HEADER
# ============================================================

st.title(
    "🫀 Vitalyn Alertia"
)

st.caption(
    "AI-powered ICU patient risk monitoring using real clinical time-series data."
)


# ============================================================
# TOP CONTROL
# ============================================================

control_left, control_right = st.columns(
    [1, 5]
)

with control_left:

    refresh_clicked = st.button(
        "↻ Refresh Predictions",
        use_container_width=True,
    )

with control_right:

    st.caption(
        f"{len(patient_ids):,} patients available"
    )


# ============================================================
# INITIAL PATIENTS
# ============================================================

# On first load, only calculate a small batch.
# This allows the UI to appear immediately.

if not st.session_state.predictions_loaded:

    initial_ids = patient_ids[
        :min(DISPLAY_COUNT, len(patient_ids))
    ]

    for patient_id in initial_ids:

        risk = get_prediction(
            patient_id
        )

        st.session_state.predictions[
            patient_id
        ] = risk

    st.session_state.predictions_loaded = True


# ============================================================
# MANUAL REFRESH
# ============================================================

if refresh_clicked:

    st.session_state.predictions = {}

    # Calculate only the first batch again.
    initial_ids = patient_ids[
        :min(DISPLAY_COUNT, len(patient_ids))
    ]

    with st.spinner(
        "Updating predictions..."
    ):

        for patient_id in initial_ids:

            risk = get_prediction(
                patient_id
            )

            st.session_state.predictions[
                patient_id
            ] = risk


# ============================================================
# CURRENT RESULTS
# ============================================================

results = []

for patient_id, risk in (
    st.session_state.predictions.items()
):

    results.append(
        {
            "patient_id": patient_id,
            "risk": risk,
            "data": get_patient_data(
                patient_id
            ),
        }
    )


# Highest risk first
results.sort(
    key=lambda x: (
        x["risk"]
        if x["risk"] is not None
        else -1
    ),
    reverse=True,
)


# ============================================================
# SUMMARY
# ============================================================

valid_results = [
    result
    for result in results
    if result["risk"] is not None
]


high_count = sum(
    result["risk"] >= CRITICAL_THRESHOLD
    for result in valid_results
)

moderate_count = sum(
    MODERATE_THRESHOLD
    <= result["risk"]
    < CRITICAL_THRESHOLD
    for result in valid_results
)

low_count = sum(
    result["risk"] < MODERATE_THRESHOLD
    for result in valid_results
)


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Patients",
        f"{len(patient_ids):,}"
    )

with col2:

    st.metric(
        "High Risk",
        high_count
    )

with col3:

    st.metric(
        "Moderate Risk",
        moderate_count
    )

with col4:

    st.metric(
        "Lower Risk",
        low_count
    )


# ============================================================
# TABS
# ============================================================

monitor_tab, analytics_tab, ai_tab = st.tabs(
    [
        "🏥 Patient Monitor",
        "📊 Analytics",
        "🤖 AI Recommendations",
    ]
)


# ============================================================
# PATIENT MONITOR
# ============================================================

with monitor_tab:

    st.subheader(
        "Live Patient Monitor"
    )

    st.caption(
        f"Showing the highest-risk patients from the "
        f"{len(patient_ids):,} available patients."
    )
    cards_col, trend_col = st.columns([2, 1])

    with cards_col:
        if not results:
            st.info(
                "No predictions available yet."
            )
        else:
            # Sort results by descending risk, placing None at the end (red → yellow → green)
            results = sorted(results, key=lambda r: (r["risk"] is None, -(r["risk"] or 0)))
            for row_start in range(0, len(results), 3):
                row = results[row_start:row_start + 3]

                columns = st.columns(
                    len(row)
                )

                for column, result in zip(
                    columns,
                    row
                ):
                    with column:
                        patient_id = result["patient_id"]
                        risk = result["risk"]
                        df = result["data"]

                        if risk is None:
                            # Simple grey card for unavailable prediction
                            card_html = f"""
                            <div style='background-color:#f0f0f0;padding:12px;border-radius:8px;margin-bottom:12px'>
                              <h4 style='margin:0'>{patient_id}</h4>
                              <p style='margin:4px 0'>Prediction unavailable</p>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
                            continue

                        # Determine background color based on risk
                        if risk >= CRITICAL_THRESHOLD:
                            bg_color = "#8B0000"  # dark red
                        elif risk >= MODERATE_THRESHOLD:
                            bg_color = "#B8860B"  # dark yellow/goldenrod
                        else:
                            bg_color = "#006400"  # dark green

                        # Build card HTML
                        card_html = f"""
                        <div style='background-color:{bg_color};padding:12px;border-radius:8px;margin-bottom:12px'>
                          <h4 style='margin:0'>{patient_id}</h4>
                          <p style='margin:4px 0'>AI Risk: {risk*100:.1f}%</p>
                        """

                        # Basic patient info
                        age = latest_value(df, "Age")
                        icu_hour = latest_value(df, "ICULOS")
                        card_html += f"<p style='margin:2px 0'>Age: {format_value(age,' yrs')} • ICU Hour: {format_value(icu_hour,'',0)}</p>"

                        # Vital metrics list
                        vitals = {
                            "Resp": ("/min", None),
                            "HR": (" bpm", 0),
                            "SBP": (" mmHg", 0),
                            "Temp": (" °C", None),
                        }
                        vitals_html = "<ul style='padding-left:20px;margin:4px 0'>"
                        for name, (unit, dec) in vitals.items():
                            val = latest_value(df, name)
                            vitals_html += f"<li>{name}: {format_value(val, unit, dec if dec is not None else 1)}</li>"
                        vitals_html += "</ul>"
                        card_html += vitals_html
                        card_html += "</div>"
                        st.markdown(card_html, unsafe_allow_html=True)

        # Duplicate card rendering block removed




    with trend_col:
        st.divider()
        st.subheader(
            "🔎 Patient Trend Explorer"
        )
        if patient_ids:
            focus_patient = st.selectbox(
                "Select patient",
                patient_ids
            )
            focus_df = get_patient_data(
                focus_patient
            )
            focus_risk = get_prediction(
                focus_patient
            )
            if focus_risk is not None:
                st.metric(
                    "Current AI Risk",
                    f"{focus_risk * 100:.1f}%"
                )
            if (
                focus_df is not None
                and not focus_df.empty
            ):
                try:
                    fig = plot_vital_trends(
                        focus_df
                    )
                    if fig is not None:
                        st.pyplot(
                            fig,
                            use_container_width=True
                        )
                except Exception as e:
                    st.warning(
                        f"Could not display trend: {e}"
                    )


    # ========================================================
    # DATASET INFO
    # ========================================================

    st.divider()

    with st.expander(
        "📊 Dataset Information"
    ):

        st.write(
            "Vitalyn Alertia uses the PhysioNet "
            "Computing in Cardiology Challenge 2019 "
            "Early Prediction of Sepsis dataset."
        )

        st.write(
            f"Patients available locally: "
            f"**{len(patient_ids):,}**"
        )

        st.write(
            "Each patient file contains hourly ICU "
            "measurements and a SepsisLabel target."
        )

        st.write(
            "The model uses physiological time-series "
            "features including respiratory rate, "
            "systolic blood pressure, heart rate, "
            "temperature, age, historical averages, "
            "and short-term trends."
        )


# ============================================================
# ANALYTICS
# ============================================================

with analytics_tab:

# Removed header and caption as per user request

    col1, col3, col4 = st.columns(3)

    with col1:

        st.metric(
            "Total Dataset",
            f"{len(patient_ids):,}"
        )



    with col3:

        if valid_results:

            avg_risk = np.mean(
                [
                    r["risk"]
                    for r in valid_results
                ]
            )

            st.metric(
                "Average Risk",
                f"{avg_risk * 100:.1f}%"
            )

        else:

            st.metric(
                "Average Risk",
                "—"
            )

    with col4:

        if valid_results:

            max_risk = max(
                r["risk"]
                for r in valid_results
            )

            st.metric(
                "Highest Risk",
                f"{max_risk * 100:.1f}%"
            )

        else:

            st.metric(
                "Highest Risk",
                "—"
            )

    st.divider()

    if valid_results:

        distribution = pd.DataFrame(
            {
                "Risk Group": [
                    "High",
                    "Moderate",
                    "Lower"
                ],
                "Patients": [
                    high_count,
                    moderate_count,
                    low_count
                ]
            }
        )

        st.subheader(
            "Risk Distribution"
        )

        st.bar_chart(
            distribution.set_index(
                "Risk Group"
            )
        )

        st.divider()

        st.subheader(
            "Evaluated Patients"
        )

        rows = []

        for result in valid_results:

            df = result["data"]

            rows.append(
                {
                    "Patient":
                        result["patient_id"],

                    "AI Risk":
                        f"{result['risk'] * 100:.1f}%",

                    "Age":
                        latest_value(
                            df,
                            "Age"
                        ),

                    "Resp":
                        latest_value(
                            df,
                            "Resp"
                        ),

                    "SBP":
                        latest_value(
                            df,
                            "SBP"
                        ),

                    "HR":
                        latest_value(
                            df,
                            "HR"
                        ),

                    "Temp":
                        latest_value(
                            df,
                            "Temp"
                        ),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No patient predictions are currently available."
        )


# ============================================================
# AI RECOMMENDATIONS
# ============================================================

with ai_tab:

    st.header(
        "🤖 AI Recommendations"
    )

    st.caption(
        "Observations generated from the current model risk "
        "and physiological measurements."
    )

    if not valid_results:

        st.info(
            "No predictions available."
        )

    else:

        for result in valid_results[:15]:

            patient_id = result[
                "patient_id"
            ]

            risk = result[
                "risk"
            ]

            df = result[
                "data"
            ]

            observations = []

            resp = latest_value(
                df,
                "Resp"
            )

            sbp = latest_value(
                df,
                "SBP"
            )

            hr = latest_value(
                df,
                "HR"
            )

            temp = latest_value(
                df,
                "Temp"
            )

            if resp is not None and resp >= 24:

                observations.append(
                    f"Respiratory rate elevated at "
                    f"{resp:.1f}/min."
                )

            if resp is not None and resp <= 8:

                observations.append(
                    f"Respiratory rate low at "
                    f"{resp:.1f}/min."
                )

            if sbp is not None and sbp < 90:

                observations.append(
                    f"Systolic blood pressure low at "
                    f"{sbp:.0f} mmHg."
                )

            if hr is not None and hr >= 100:

                observations.append(
                    f"Heart rate elevated at "
                    f"{hr:.0f} bpm."
                )

            if hr is not None and hr < 50:

                observations.append(
                    f"Heart rate low at "
                    f"{hr:.0f} bpm."
                )

            if temp is not None and temp >= 38:

                observations.append(
                    f"Temperature elevated at "
                    f"{temp:.1f} °C."
                )

            if temp is not None and temp < 36:

                observations.append(
                    f"Temperature low at "
                    f"{temp:.1f} °C."
                )

            with st.container(
                border=True
            ):

                left, right = st.columns(
                    [3, 1]
                )

                with left:

                    st.markdown(
                        f"### {patient_id}"
                    )

                with right:

                    st.metric(
                        "AI Risk",
                        f"{risk * 100:.1f}%"
                    )

                if observations:

                    for observation in observations:

                        st.write(
                            f"• {observation}"
                        )

                else:

                    st.write(
                        "No major abnormality identified "
                        "from the displayed vital signs."
                    )