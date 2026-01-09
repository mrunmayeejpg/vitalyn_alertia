🩺 Vitalyn Alertia
AI-Driven ICU Patient Risk Monitoring Dashboard

Hackathon Project – VORTEX 2026

🚨 Problem Statement

In many hospitals—especially rural and understaffed regions—a single doctor or nurse is responsible for dozens of patients.
Critical deterioration often goes unnoticed until it is too late.

Sepsis, respiratory failure, and sudden hypotension can escalate within minutes.

👉 Vitalyn Alertia was built to detect early warning signs, prioritize critical patients, and support doctors with AI-powered risk insights—not replace them.

💡 Solution Overview

Vitalyn Alertia is a real-time ICU command dashboard that continuously:

Monitors patient vitals

Calculates qSOFA scores

Predicts short-term deterioration risk using an ML-based model

Visually ranks patients so doctors act before emergencies occur

⚠️ This system is strictly a screening & risk-support tool — not a diagnostic system.

🧠 Core Features
🔹 Real-Time Patient Monitoring

Respiratory Rate (RR)

Blood Pressure (BP)

Heart Rate (HR)

Temperature

Mental Status

Vitals update every 4 seconds via live simulation.

🔹 qSOFA Rule-Based Screening

Automatically computes qSOFA score using:

RR ≥ 22

SBP ≤ 100

Altered mental status

Provides clinically interpretable risk flags.

🔹 Machine Learning–Inspired AI Risk Model (Key Feature)

A lightweight ML-style risk model estimates probability of deterioration using:

Current vitals

qSOFA score

Normalized physiological thresholds

Output:

A continuous risk score (0–1)

Short-term deterioration predictions (30 min – 6 hrs)

Patients are auto-classified into:

🔴 Critical

🟡 Moderate

🟢 Stable

🔹 Live ICU Command Board

Patients auto-sorted by risk

Critical patients float to the top

Smooth UI animations for rapid attention

🔹 Doctor-Only Secure Login

Restricted dashboard access

Doctor selection for accountability

🔹 Patient Trend Analytics

Time-series plots of:

RR

BP

AI Risk

Helps doctors observe worsening trends, not just static values

🏥 Why This Matters (Especially for Villages)

In many villages:

There are too few doctors

ICU expertise is limited

Delays cost lives

💔 Vitalyn Alertia acts as a digital second pair of eyes, ensuring:

No critical patient is silently ignored

Doctors focus attention where it’s needed most

Lives are saved through early escalation

🛠️ Tech Stack

Frontend & Dashboard: Streamlit

Data Processing: Pandas, NumPy

Visualization: Matplotlib

AI Logic: Rule-based + ML-inspired risk scoring

Real-Time Simulation: Streamlit Auto Refresh

Language: Python