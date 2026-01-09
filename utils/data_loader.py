import pandas as pd
from data.patient_scenarios import PATIENT_SCENARIOS


def list_patients():
    return list(PATIENT_SCENARIOS.keys())


def load_patient(patient_name):
    return pd.DataFrame(PATIENT_SCENARIOS[patient_name])
