# vortex/utils/db_utils.py
import sqlite3
import pandas as pd

DB_FILE = "vortex/patient_logs.db"

def create_table():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS patient_logs (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            patient_id TEXT,
            hospital TEXT,
            doctor TEXT,
            age INTEGER,
            resp REAL,
            bp REAL,
            hr REAL,
            temp REAL,
            mental TEXT,
            qsofa INTEGER,
            ai_risk REAL,
            risk_level TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_patient(patient):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO patient_logs (
            patient_id, hospital, doctor, age, resp, bp, hr, temp,
            mental, qsofa, ai_risk, risk_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient["patient_id"], patient["hospital"], patient["doctor"], patient["age"],
        patient["resp"], patient["bp"], patient["hr"], patient["temp"],
        patient["mental"], patient["qSOFA"], patient["AI_Risk"], patient["Risk_Level"]
    ))
    conn.commit()
    conn.close()

def fetch_logs():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM patient_logs ORDER BY timestamp DESC", conn)
    conn.close()
    return df
