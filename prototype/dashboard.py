import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Offshore Wind Turbine Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 Offshore Wind Turbine – Digital Twin Dashboard")

# ==============================
# DATA PATHS
# ==============================
ANOMALY_PATH = "data/processed/anomaly_with_root_cause.csv"
HEALTH_PATH = "data/processed/health_index.csv"
RUL_PATH = "data/processed/realtime_rul.csv"

# ==============================
# SAFE TIMESTAMP HANDLER
# ==============================
def normalize_timestamp(df):
    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        return df, "time_stamp"
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.rename(columns={"timestamp": "time_stamp"})
        return df, "time_stamp"
    else:
        raise ValueError(f"No timestamp column found. Columns: {df.columns.tolist()}")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    anomaly_df = pd.read_csv(ANOMALY_PATH)
    health_df = pd.read_csv(HEALTH_PATH)
    rul_df = pd.read_csv(RUL_PATH)

    anomaly_df, _ = normalize_timestamp(anomaly_df)
    health_df, _ = normalize_timestamp(health_df)
    rul_df, _ = normalize_timestamp(rul_df)

    return anomaly_df, health_df, rul_df


if not (os.path.exists(ANOMALY_PATH) and os.path.exists(HEALTH_PATH) and os.path.exists(RUL_PATH)):
    st.error("❌ One or more processed data files are missing.")
    st.stop()

anomaly_df, health_df, rul_df = load_data()

# Drop rows with invalid timestamps
anomaly_df = anomaly_df.dropna(subset=["time_stamp"])
health_df = health_df.dropna(subset=["time_stamp"])
rul_df = rul_df.dropna(subset=["time_stamp"])

# ==============================
# SIDEBAR CONTROLS
# ==============================
st.sidebar.header("⚙️ Controls")

start_date = st.sidebar.date_input(
    "Start Date", value=health_df["time_stamp"].min().date()
)
end_date = st.sidebar.date_input(
    "End Date", value=health_df["time_stamp"].max().date()
)

health_df_f = health_df[
    (health_df["time_stamp"].dt.date >= start_date) &
    (health_df["time_stamp"].dt.date <= end_date)
]

rul_df_f = rul_df[
    (rul_df["time_stamp"].dt.date >= start_date) &
    (rul_df["time_stamp"].dt.date <= end_date)
]

anomaly_df_f = anomaly_df[
    (anomaly_df["time_stamp"].dt.date >= start_date) &
    (anomaly_df["time_stamp"].dt.date <= end_date)
]

# ==============================
# KPI METRICS (SAFE)
# ==============================
latest_health = health_df_f["health_index"].dropna().iloc[-1] if not health_df_f.empty else 0
latest_rul = rul_df_f["RealTime_RUL_hours"].dropna().iloc[-1] if not rul_df_f.empty else 0
active_faults = anomaly_df_f["is_anomaly"].sum() if "is_anomaly" in anomaly_df_f else len(anomaly_df_f)

risk_level = "LOW"
if latest_rul < 100:
    risk_level = "CRITICAL"
elif latest_rul < 250:
    risk_level = "HIGH"
elif latest_rul < 400:
    risk_level = "MEDIUM"

col1, col2, col3, col4 = st.columns(4)
col1.metric("⚡ Turbine Health", f"{latest_health:.3f}")
col2.metric("⏳ Real-Time RUL (hrs)", f"{latest_rul:.1f}")
col3.metric("🚨 Active Anomalies", int(active_faults))
col4.metric("⚠️ Risk Level", risk_level)

st.divider()

# ==============================
# HEALTH TREND
# ==============================
st.subheader("📈 Health Degradation Trend")

fig_health = px.line(
    health_df_f,
    x="time_stamp",
    y="health_index",
    title="Health Index Over Time"
)

st.plotly_chart(fig_health, use_container_width=True)

# ==============================
# RUL TREND
# ==============================
st.subheader("⏳ Real-Time Remaining Useful Life")

fig_rul = px.line(
    rul_df_f,
    x="time_stamp",
    y="RealTime_RUL_hours",
    title="Real-Time RUL (Hours)"
)

st.plotly_chart(fig_rul, use_container_width=True)

# ==============================
# FAULT & RCA ANALYSIS
# ==============================
st.subheader("🛠 Fault Distribution by Subsystem (RCA)")

if "root_cause_physical" in anomaly_df_f.columns:
    rca_counts = anomaly_df_f["root_cause_physical"].value_counts().reset_index()
    rca_counts.columns = ["Subsystem", "Fault Count"]

    fig_rca = px.bar(
        rca_counts,
        x="Subsystem",
        y="Fault Count",
        title="Fault Count per Subsystem"
    )

    st.plotly_chart(fig_rca, use_container_width=True)
else:
    st.warning("RCA column not found in anomaly file.")

# ==============================
# ANOMALY TABLE
# ==============================
st.subheader("📋 Recent Anomalies")

display_cols = [c for c in anomaly_df_f.columns if "Unnamed" not in c]
st.dataframe(anomaly_df_f[display_cols].tail(20), use_container_width=True)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown(
    "✅ **Digital Twin includes:** Anomaly Detection, Root Cause Analysis, "
    "Health Index Estimation, and Real-Time RUL Prediction."
)
