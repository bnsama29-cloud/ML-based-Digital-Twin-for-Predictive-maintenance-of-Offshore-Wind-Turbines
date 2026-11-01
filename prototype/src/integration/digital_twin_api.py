from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
import os
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
DATA_TELEMETRY = "data/processed/44_processed.csv"
DATA_RUL = "data/processed/realtime_rul.csv"
DATA_ANOMALIES = "data/processed/anomaly_with_root_cause.csv"
DATA_MAINTENANCE = "data/maintenance_schedule.csv"

# -----------------------------
# INIT FASTAPI APP
# -----------------------------
app = FastAPI(
    title="Wind Turbine Digital Twin API",
    description="Backend powering Unity Digital Twin Visualization",
    version="2.0"
)

# -----------------------------
# ENABLE CORS FOR UNITY
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow Unity Editor + builds
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# RESPONSE MODELS
# -----------------------------

class TelemetryOut(BaseModel):
    timestamp: str
    rpm: float
    wind_speed: float
    power_output: float
    temp_gearbox: float
    temp_generator: float
    health_index: float
    rul_hours: float


class AnomalyOut(BaseModel):
    timestamp: str
    is_anomaly: bool
    sensors: str
    subsystem: str


class MaintenanceItem(BaseModel):
    Subsystem: str
    Effective_RUL_hrs: float
    Priority_Score: float
    Recommended_Action: str
    Predicted_Maintenance_Due: str


# -----------------------------
# ROUTES
# -----------------------------

@app.get("/")
def root():
    return {"status": "Digital Twin API Running", "version": "2.0"}


# ------------------------------------------------------------
# 1️⃣  REAL-TIME Telemetry Endpoint (Unity calls this every ~0.2s)
# ------------------------------------------------------------
@app.get("/api/telemetry", response_model=TelemetryOut)
def get_realtime_telemetry():

    df = pd.read_csv(DATA_TELEMETRY)
    rul_df = pd.read_csv(DATA_RUL)

    latest = df.iloc[-1]
    latest_rul = rul_df.iloc[-1]

    return TelemetryOut(
        timestamp=str(latest["time_stamp"]),
        rpm=float(latest.get("rpm", 12)),
        wind_speed=float(latest.get("wind_speed", 7)),
        power_output=float(latest.get("power_output", 1200)),
        temp_gearbox=float(latest.get("temp_gearbox", 45)),
        temp_generator=float(latest.get("temp_generator", 50)),
        health_index=float(latest_rul.get("health_index", 1.0)),
        rul_hours=float(latest_rul.get("RealTime_RUL_hours", 200)),
    )


# ------------------------------------------------------------
# 2️⃣  HISTORICAL TREND DATA (Useful for dashboards)
# ------------------------------------------------------------
@app.get("/api/history")
def get_history(n: int = 500):
    df = pd.read_csv(DATA_TELEMETRY)
    return df.tail(n).to_dict(orient="records")


# ------------------------------------------------------------
# 3️⃣  Real-Time RUL + Health
# ------------------------------------------------------------
@app.get("/api/rul")
def get_rul():
    df = pd.read_csv(DATA_RUL)
    return df.tail(200).to_dict(orient="records")


# ------------------------------------------------------------
# 4️⃣  Anomalies + RCA Output
# ------------------------------------------------------------
@app.get("/api/anomalies", response_model=list[AnomalyOut])
def get_anomalies():
    if not os.path.exists(DATA_ANOMALIES):
        return []

    df = pd.read_csv(DATA_ANOMALIES)

    return [
        AnomalyOut(
            timestamp=str(r["timestamp"]),
            is_anomaly=bool(r["is_anomaly"]),
            sensors=r["fault_sensors"],
            subsystem=r["root_cause"]
        )
        for _, r in df.iterrows()
    ]


# ------------------------------------------------------------
# 5️⃣ Predictive Maintenance Schedule
# ------------------------------------------------------------
@app.get("/api/maintenance", response_model=list[MaintenanceItem])
def get_maintenance():

    if not os.path.exists(DATA_MAINTENANCE):
        return []

    df = pd.read_csv(DATA_MAINTENANCE)

    return [
        MaintenanceItem(
            Subsystem=row["Subsystem"],
            Effective_RUL_hrs=float(row["Effective RUL (hrs)"]),
            Priority_Score=float(row["Priority Score"]),
            Recommended_Action=row["Recommended Action"],
            Predicted_Maintenance_Due=str(row["Predicted Maintenance Due"])
        )
        for _, row in df.iterrows()
    ]
