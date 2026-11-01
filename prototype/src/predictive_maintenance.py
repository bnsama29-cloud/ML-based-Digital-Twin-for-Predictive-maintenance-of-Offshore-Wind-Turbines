#!/usr/bin/env python3
"""
predictive_maintenance.py

Generates:
 - data/processed/maintenance_schedule.csv

Uses:
 - realtime RUL
 - anomaly + RCA output
 - subsystem criticality
"""

import pandas as pd
import numpy as np
import os

# ==============================
# FILE PATHS
# ==============================
RUL_PATH = "data/processed/realtime_rul.csv"
ANOMALY_PATH = "data/processed/anomaly_with_root_cause.csv"
OUT_PATH = "data/processed/maintenance_schedule.csv"

# ==============================
# SUBSYSTEM CRITICALITY
# ==============================
DEFAULT_CRITICALITY = {
    "GEARBOX": 1.0,
    "GENERATOR": 1.0,
    "POWER_ELECTRONICS": 0.9,
    "SHAFT": 0.9,
    "ROTOR": 0.7,
    "PITCH": 0.7,
    "YAW": 0.6,
    "TOWER": 0.6,
    "GRID": 0.5,
    "ENVIRONMENT": 0.2,
    "UNKNOWN": 0.5
}

# ==============================
# HELPERS
# ==============================
def normalize_timestamp(df):
    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.rename(columns={"timestamp": "time_stamp"})
    else:
        raise ValueError(f"No timestamp column found. Columns: {df.columns.tolist()}")
    return df


def detect_subsystem_column(df):
    candidates = [
        "root_cause",
        "RCA",
        "root_cause_combined",
        "subsystem",
        "physical_subsystem",
        "pred_subsystem",
        "fault_subsystem"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ==============================
# LOAD DATA
# ==============================
if not os.path.exists(RUL_PATH):
    raise FileNotFoundError(f"Missing file: {RUL_PATH}")

if not os.path.exists(ANOMALY_PATH):
    raise FileNotFoundError(f"Missing file: {ANOMALY_PATH}")

rul_df = pd.read_csv(RUL_PATH)
anom_df = pd.read_csv(ANOMALY_PATH)

rul_df = normalize_timestamp(rul_df)
anom_df = normalize_timestamp(anom_df)

rul_df = rul_df.dropna(subset=["time_stamp"])
anom_df = anom_df.dropna(subset=["time_stamp"])

# ==============================
# RUL COLUMN SAFETY
# ==============================
if "RealTime_RUL_hours" not in rul_df.columns:
    alt = [c for c in rul_df.columns if "rul" in c.lower()]
    if not alt:
        raise ValueError("No RUL column found in realtime_rul.csv")
    rul_df = rul_df.rename(columns={alt[0]: "RealTime_RUL_hours"})

# ==============================
# SUBSYSTEM COLUMN SAFETY
# ==============================
subsystem_col = detect_subsystem_column(anom_df)

if subsystem_col is None:
    anom_df["pred_subsystem"] = "UNKNOWN"
    subsystem_col = "pred_subsystem"

# ==============================
# BASE RUL & TIME
# ==============================
latest_row = rul_df.sort_values("time_stamp").iloc[-1]
now_ts = latest_row["time_stamp"]
base_rul = float(latest_row["RealTime_RUL_hours"])

MAX_RUL = max(1.0, base_rul)

# ==============================
# RECENT ANOMALIES (30 DAYS)
# ==============================
LOOKBACK_DAYS = 30
time_cut = now_ts - pd.Timedelta(days=LOOKBACK_DAYS)

recent_anom = anom_df[anom_df["time_stamp"] >= time_cut]

anom_stats = (
    recent_anom
    .groupby(subsystem_col)
    .agg(
        recent_anom_count=("time_stamp", "count"),
        last_anomaly_time=("time_stamp", "max")
    )
    .reset_index()
)

# ==============================
# MAINTENANCE SCHEDULING
# ==============================
records = []

for subsystem, criticality in DEFAULT_CRITICALITY.items():

    row = anom_stats[anom_stats[subsystem_col] == subsystem]

    if len(row) == 0:
        anom_count = 0
        recency_factor = 0.1
    else:
        anom_count = int(row["recent_anom_count"].iloc[0])

        last_time = row["last_anomaly_time"].iloc[0]
        recency_hours = max(
            1.0,
            (now_ts - last_time).total_seconds() / 3600
        )

        recency_factor = np.exp(-recency_hours / 72)  # 3-day decay

    # ✅ FIXED SUBSYSTEM-SPECIFIC DEGRADATION
    degradation = (
        0.15 * anom_count +
        0.50 * recency_factor +
        0.35 * criticality
    )

    degradation = np.clip(degradation, 0.05, 0.9)

    effective_rul = base_rul * (1.0 - degradation)

    # ✅ PRIORITY SCORE
    score_rul = 1.0 - (effective_rul / MAX_RUL)
    score_anom = min(1.0, anom_count / 12.0)

    priority = (
        0.5 * score_rul +
        0.3 * score_anom +
        0.2 * criticality
    )

    predicted_due = now_ts + pd.Timedelta(hours=effective_rul)

    # ✅ ACTION WINDOWS (FIXED)
    if effective_rul < 24:
        action = "Emergency Shutdown & Repair"
    elif effective_rul < 72:
        action = "Immediate Maintenance (48–72 hrs)"
    elif effective_rul < 168:
        action = "High Priority Maintenance (1 week)"
    elif effective_rul < 500:
        action = "Schedule Maintenance (2–3 weeks)"
    else:
        action = "Routine Monitoring Only"

    records.append({
        "Subsystem": subsystem,
        "Base RUL (hrs)": round(base_rul, 2),
        "Effective RUL (hrs)": round(effective_rul, 2),
        "Recent Anomalies": anom_count,
        "Criticality": round(criticality, 2),
        "Recency Factor": round(recency_factor, 3),
        "Priority Score": round(priority, 4),
        "Predicted Maintenance Due": predicted_due,
        "Recommended Action": action
    })

# ==============================
# SAVE OUTPUT
# ==============================
sched_df = pd.DataFrame(records).sort_values(
    "Priority Score", ascending=False
)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
sched_df.to_csv(OUT_PATH, index=False)

print("✅ Predictive maintenance schedule created successfully!")
print("📁 Saved to:", OUT_PATH)
