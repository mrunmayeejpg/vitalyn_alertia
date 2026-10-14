# Vitalyn Alertia

## Explainable AI-Based Early Sepsis Risk Monitoring System

Vitalyn Alertia is an AI-powered clinical decision-support prototype designed to estimate the risk of sepsis from ICU patient time-series data.

The system uses real clinical data from the **PhysioNet Computing in Cardiology Challenge 2019: Early Prediction of Sepsis from Clinical Data**.

> **Important:** Vitalyn Alertia is a research/educational prototype. Its predictions are risk estimates and do not constitute a medical diagnosis or treatment recommendation.

---

## Key Features

- Real ICU time-series data
- Hour-by-hour patient monitoring
- Early sepsis risk prediction
- Multiple machine-learning models
- Patient-level train/test separation
- Physiological trend features
- Risk-level classification
- Interactive Streamlit dashboard
- Explainable physiological indicators
- SepsisLabel-based evaluation

---

## Dataset

Vitalyn Alertia uses:

**PhysioNet Computing in Cardiology Challenge 2019**

The dataset contains hourly ICU observations for individual patients.

Each patient is represented by a `.psv` file.

Important variables include:

### Vital Signs

- Heart Rate (`HR`)
- Oxygen Saturation (`O2Sat`)
- Temperature (`Temp`)
- Systolic Blood Pressure (`SBP`)
- Mean Arterial Pressure (`MAP`)
- Diastolic Blood Pressure (`DBP`)
- Respiratory Rate (`Resp`)
- End-Tidal CO2 (`EtCO2`)

### Laboratory Measurements

- Lactate
- WBC
- Creatinine
- Glucose
- Platelets
- Hgb
- Hct
- BUN
- Bilirubin
- Potassium
- Sodium
- Calcium
- and other laboratory variables

### Patient Information

- Age
- Gender
- Unit1
- Unit2
- Hospital admission time
- ICU length of stay (`ICULOS`)

### Target

`SepsisLabel`

- `0` → No sepsis label
- `1` → Sepsis label

---

## Machine Learning Features

The current model uses the following features:

### Current physiological measurements

- Respiratory rate
- Systolic blood pressure
- Heart rate
- Temperature
- Age

### Historical averages

- Average respiratory rate
- Average systolic blood pressure
- Average heart rate
- Average temperature

### Recent trends

- Respiratory-rate trend
- Blood-pressure trend
- Heart-rate trend
- Temperature trend

The model therefore uses both the patient's current physiological state and recent changes.

---

## Machine Learning Models

The training pipeline compares:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

Class imbalance is handled during training.

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

The final model is selected based on F1-score.

---

## Project Structure

```text
vitalyn_alertia/
│
├── data/
│   └── raw/
│       └── training_setA/
│           ├── p000001.psv
│           ├── p000002.psv
│           └── ...
│
├── logic/
│   ├── ai_predict.py
│   └── explainability.py
│
├── ml/
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   └── saved_models/
│       └── sepsis_model.pkl
│
├── utils/
│   └── data_loader.py
│
├── visualization/
│   └── charts.py
│
├── app.py
├── requirements.txt
└── README.md

---
## Recent Updates

- Increased the number of patients evaluated on app start from **30** to **300** to provide a broader view of risk assessments.
- Made feature extraction tolerant to missing columns and age values, defaulting to zero where appropriate, which raised the evaluated patient count well above the previous ~29.
- Updated risk calculation validation tests to cover normal, borderline, and extreme vital signs, missing inputs, and score boundaries.
- Documented the limitations of the qSOFA‑inspired risk approach and clarified that this is a research prototype, not a clinical diagnostic tool.
- Added dataset source information and detailed feature list.

These changes improve the dashboard's usability and reliability for exploratory analyses.