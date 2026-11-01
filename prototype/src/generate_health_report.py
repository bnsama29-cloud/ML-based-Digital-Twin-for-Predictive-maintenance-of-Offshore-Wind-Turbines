#!/usr/bin/env python3
"""
generate_health_report.py

Produces:
 - data/processed/health_report.html
 - data/processed/health_report_summary.csv
 - data/processed/health_plots/*.png
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- CONFIG ----------
ANOMALY_PATH = "data/processed/anomaly_with_root_cause.csv"
HEALTH_PATH = "data/processed/health_index.csv"
RUL_PATH = "data/processed/realtime_rul.csv"

OUT_HTML = "data/processed/health_report.html"
OUT_SUMMARY = "data/processed/health_report_summary.csv"
PLOTS_DIR = "data/processed/health_plots"

# ensure directories
os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------- HELPERS ----------
def normalize_timestamp(df, name):
    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    elif "timestamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"], errors="ignore")
    else:
        raise ValueError(f"No timestamp column found in {name}")
    return df

# ---------- LOAD ----------
if not os.path.exists(HEALTH_PATH):
    raise FileNotFoundError(f"{HEALTH_PATH} missing. Run build_health_index.py first.")
health_df = pd.read_csv(HEALTH_PATH)
health_df = normalize_timestamp(health_df, HEALTH_PATH)
health_df = health_df.sort_values("time_stamp").dropna(subset=["time_stamp"])

if os.path.exists(RUL_PATH):
    rul_df = pd.read_csv(RUL_PATH)
    rul_df = normalize_timestamp(rul_df, RUL_PATH)
    rul_df = rul_df.sort_values("time_stamp").dropna(subset=["time_stamp"])
else:
    rul_df = None

if os.path.exists(ANOMALY_PATH):
    anom_df = pd.read_csv(ANOMALY_PATH)
    anom_df = normalize_timestamp(anom_df, ANOMALY_PATH)
else:
    anom_df = pd.DataFrame()

# ---------- SUMMARY METRICS ----------
latest_time = health_df["time_stamp"].max()
latest_health = float(health_df["health_index"].iloc[-1])
health_trend = health_df["health_index"].iloc[-1] - health_df["health_index"].iloc[max(0, len(health_df)-50)]
avg_health_30d = health_df[health_df["time_stamp"] >= latest_time - pd.Timedelta(days=30)]["health_index"].mean()

summary = {
    "report_generated_at": datetime.utcnow().isoformat(),
    "latest_time": latest_time,
    "latest_health": latest_health,
    "health_change_last_50_samples": float(health_trend),
    "avg_health_last_30d": float(avg_health_30d) if not np.isnan(avg_health_30d) else None,
    "total_anomalies": len(anom_df),
}

if rul_df is not None:
    latest_rul = float(rul_df["RealTime_RUL_hours"].dropna().iloc[-1])
    summary["latest_rul_hours"] = latest_rul

# ---------- ANOMALY BREAKDOWN ----------
if not anom_df.empty:
    # detect subsystem column
    candidates = ["root_cause", "root_cause_physical", "RCA", "subsystem", "pred_subsystem"]
    sub_col = next((c for c in candidates if c in anom_df.columns), None)
    if sub_col:
        breakdown = anom_df[sub_col].value_counts().head(10)
    else:
        breakdown = anom_df.columns.value_counts().head(10)
else:
    breakdown = pd.Series(dtype=int)

# ---------- PLOTS ----------
# 1) health over time
plt.figure(figsize=(10,4))
plt.plot(health_df["time_stamp"], health_df["health_index"], label="Health Index")
plt.xlabel("Time")
plt.ylabel("Health Index")
plt.title("Health Index Over Time")
plt.grid(True)
plt.tight_layout()
p1 = os.path.join(PLOTS_DIR, "health_index.png")
plt.savefig(p1)
plt.close()

# 2) RUL over time (if available)
p2 = None
if rul_df is not None:
    plt.figure(figsize=(10,4))
    plt.plot(rul_df["time_stamp"], rul_df["RealTime_RUL_hours"], label="RUL (hours)")
    plt.xlabel("Time")
    plt.ylabel("RUL (hours)")
    plt.title("Real-Time RUL Over Time")
    plt.grid(True)
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "rul.png")
    plt.savefig(p2)
    plt.close()

# 3) Anomalies per subsystem bar
p3 = None
if not anom_df.empty and sub_col:
    fig = breakdown.plot(kind="bar", figsize=(10,4), title="Top Fault Subsystems")
    p3 = os.path.join(PLOTS_DIR, "fault_subsystems.png")
    fig.figure.savefig(p3)
    plt.close()

# ---------- SAVE SUMMARY CSV ----------
summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUT_SUMMARY, index=False)

# ---------- BUILD HTML REPORT ----------
html_parts = []
html_parts.append(f"<h1>Wind Turbine Health Report</h1>")
html_parts.append(f"<p>Generated at (UTC): {summary['report_generated_at']}</p>")
html_parts.append("<h2>Key Metrics</h2><ul>")
for k, v in summary.items():
    html_parts.append(f"<li><b>{k}</b>: {v}</li>")
html_parts.append("</ul>")

html_parts.append("<h2>Plots</h2>")
html_parts.append(f"<h3>Health Index Over Time</h3><img src='{os.path.basename(p1)}' width='800'>")
if p2:
    html_parts.append(f"<h3>RUL Over Time</h3><img src='{os.path.basename(p2)}' width='800'>")
if p3:
    html_parts.append(f"<h3>Top Fault Subsystems</h3><img src='{os.path.basename(p3)}' width='800'>")

# write a self-contained folder with images + html
report_dir = os.path.dirname(OUT_HTML)
assets_dir = os.path.join(report_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)
# copy plot files into assets
import shutil
shutil.copy(p1, os.path.join(assets_dir, os.path.basename(p1)))
if p2:
    shutil.copy(p2, os.path.join(assets_dir, os.path.basename(p2)))
if p3:
    shutil.copy(p3, os.path.join(assets_dir, os.path.basename(p3)))

# build HTML referencing local assets
html = "<html><head><title>Health Report</title></head><body>"
html += "".join(html_parts)
html += "</body></html>"

with open(OUT_HTML, "w") as f:
    f.write(html)

print("✅ Health report generated:")
print(" - HTML:", OUT_HTML)
print(" - Summary CSV:", OUT_SUMMARY)
print(" - Plots in:", PLOTS_DIR)
