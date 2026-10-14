"""
Data loading utilities for Vitalyn Alertia.

Loads real PhysioNet Challenge 2019 ICU patient data.
"""

from pathlib import Path

import pandas as pd


DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "training_setA"
)


def list_patients():
    """
    Return the available PhysioNet patient IDs.

    Returns
    -------
    list[str]
        Patient IDs such as p000001, p000002, ...
    """

    if not DATA_DIR.exists():
        return []

    patient_files = sorted(DATA_DIR.glob("*.psv"))

    return [
        file.stem
        for file in patient_files
    ]


def load_patient(patient_id):
    """
    Load one patient's complete ICU time-series data.

    Parameters
    ----------
    patient_id : str
        Patient ID, e.g. 'p000001'.

    Returns
    -------
    pandas.DataFrame
        Hourly ICU measurements for the patient.
    """

    if not patient_id:
        raise ValueError("Patient ID cannot be empty.")

    patient_id = str(patient_id)

    if not patient_id.endswith(".psv"):
        filename = f"{patient_id}.psv"
    else:
        filename = patient_id

    patient_path = DATA_DIR / filename

    if not patient_path.exists():
        raise FileNotFoundError(
            f"Patient file not found: {patient_path}"
        )

    df = pd.read_csv(
        patient_path,
        sep="|"
    )

    if df.empty:
        raise ValueError(
            f"Patient file contains no data: {patient_path}"
        )

    return df


def load_patients(patient_ids):
    """
    Load multiple patient files.

    Parameters
    ----------
    patient_ids : iterable[str]
        Patient IDs to load.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from patient ID to patient DataFrame.
    """

    patients = {}

    for patient_id in patient_ids:
        try:
            patients[patient_id] = load_patient(patient_id)
        except (FileNotFoundError, ValueError) as error:
            print(
                f"Warning: could not load {patient_id}: {error}"
            )

    return patients


def get_patient_summary(patient_id):
    """
    Return basic metadata and dataset information for a patient.
    """

    df = load_patient(patient_id)

    age = None
    if "Age" in df.columns:
        age_values = pd.to_numeric(
            df["Age"],
            errors="coerce"
        ).dropna()

        if not age_values.empty:
            age = float(age_values.iloc[0])

    sepsis_label = None
    if "SepsisLabel" in df.columns:
        labels = pd.to_numeric(
            df["SepsisLabel"],
            errors="coerce"
        ).dropna()

        if not labels.empty:
            sepsis_label = int(labels.max())

    return {
        "patient_id": patient_id,
        "age": age,
        "hours": len(df),
        "sepsis_label": sepsis_label,
    }