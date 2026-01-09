"""
Synthetic patient vital sign scenarios for qSOFA-based screening.
All data is fictional and for demonstration purposes only.
"""

patient_stable = [
    {"time": "08:00", "resp_rate": 16, "systolic_bp": 124, "mental_status": "Alert"},
    {"time": "10:00", "resp_rate": 17, "systolic_bp": 122, "mental_status": "Alert"},
    {"time": "12:00", "resp_rate": 16, "systolic_bp": 120, "mental_status": "Alert"},
]

patient_deteriorating = [
    {"time": "08:00", "resp_rate": 18, "systolic_bp": 112, "mental_status": "Alert"},
    {"time": "10:00", "resp_rate": 23, "systolic_bp": 108, "mental_status": "Alert"},
    {"time": "12:00", "resp_rate": 25, "systolic_bp": 98, "mental_status": "Alert"},
    {"time": "14:00", "resp_rate": 28, "systolic_bp": 92, "mental_status": "Confused"},
]

patient_fluctuating = [
    {"time": "09:00", "resp_rate": 21, "systolic_bp": 104, "mental_status": "Alert"},
    {"time": "11:00", "resp_rate": 22, "systolic_bp": 102, "mental_status": "Alert"},
    {"time": "13:00", "resp_rate": 24, "systolic_bp": 101, "mental_status": "Alert"},
]

PATIENT_SCENARIOS = {
    "Stable Patient": patient_stable,
    "Deteriorating Patient": patient_deteriorating,
    "Fluctuating Patient": patient_fluctuating,
}
