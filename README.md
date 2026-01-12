🩺 Vitalyn Alertia

Live AI-based Patient Risk Monitoring Dashboard

Vitalyn Alertia” combines ‘Vitalyn’ (from ‘vital’ + ‘lyn’, symbolizing life and care) and ‘Alertia’ (from ‘alert’), representing a system that continuously monitors patient vitals and alerts medical staff in real-time for timely intervention

Vitalyn Alertia is a real-time patient monitoring system designed to help hospitals prioritize patients based on their vital signs and predicted risk levels using AI-inspired calculations. It combines dynamic dashboards, live simulations, and risk scoring to provide actionable insights to doctors instantly.


🚀 Key Features

Real-Time Patient Monitoring

Displays vital signs (RR, BP, HR, Temp) for multiple patients.

Patients dynamically reorder based on risk, simulating real-world triage.

AI-Powered Risk Assessment

Each patient’s risk score is calculated using vital signs and mental status.

Risk levels are categorized as Critical, Moderate, or Stable.

Top 5 critical patients trigger alerts and are highlighted visually.

Interactive Analytics

Click on a patient card to see historical trends of vitals and AI risk score.

Graphs dynamically show how patient conditions change over time.

Doctor-Friendly UI

Green background and clear visual cues.

Color-coded patient cards for quick understanding.

Login system for secure access.


🧠 AI & ML Implementation
1. Rule-Based AI Risk Score (Current Implementation)

Currently, Vitalyn Alertia uses a rule-based AI approach, inspired by clinical risk scoring:

Step 1: qSOFA Score

qSOFA (quick Sequential Organ Failure Assessment) score is calculated based on 3 parameters:

Respiration Rate (RR ≥ 22 → +1)

Blood Pressure (BP ≤ 100 → +1)

Mental Status Confused → +1

The qSOFA score ranges from 0 to 3, which reflects the severity of potential deterioration.

Step 2: AI Risk Calculation

The AI risk score combines normalized vitals and qSOFA score
This produces a value between 0 and 1.

Thresholds for categorization:

Critical: Top 5 AI_Risk patients

Moderate: AI_Risk > 0.45

Stable: AI_Risk ≤ 0.45

This is essentially a lightweight predictive model, capable of real-time patient triage.

2. Real-Time Simulation

Patient vitals are updated every 4 seconds.

Risk scores are recalculated live.

Patient cards move dynamically to reflect changes in risk-based priority, giving doctors a visual sense of urgency.


💡 Potential ML Extensions

The current system uses a rule-based scoring mechanism, but it can be extended into a full ML-based predictive dashboard:

Predictive Modeling

Collect historical patient data to train models (Logistic Regression, Random Forest, Gradient Boosting, or Neural Networks).

Predict risk of deterioration, ICU admission, or need for emergency intervention.

Time-Series Forecasting

Use sequential models (LSTM, GRU) to predict vital trends and future risk scores.

Multi-Feature Integration

Include lab test results, imaging, demographics, and comorbidities to improve predictive accuracy.

Alert Optimization

Automatically trigger alerts via email/SMS for high-risk patients.

Prioritize doctors’ actions based on predicted severity.

Hospital-Scale Deployment

Multi-hospital support to track hundreds of patients.

Aggregate AI predictions for hospital resource planning.


💻 Installation & Usage
1. Clone the Repository
git clone https://github.com/mrunmayeejpg/vitalyn_alertia.git
cd vitalyn_alertia

2. Create a Virtual Environment (Optional)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Dashboard
streamlit run app.py

5. Login

Username: admin

Password: admin123


📊 Visual Features

Dynamic Patient Cards: Color-coded, move according to risk.

Trend Graphs: Larger, clean graphs showing vital trends and AI risk.

Alerts: Horizontal bars highlight patients needing immediate attention.

Doctor Panel: Shows on-duty doctor and logout option.

