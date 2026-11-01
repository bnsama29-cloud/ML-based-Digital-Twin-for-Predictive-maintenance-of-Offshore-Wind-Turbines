import pandas as pd
import numpy as np
import os

DATA_PATH = "data/processed/44_processed.csv"
FAILURE_LOG = "data/failure_log.csv"
OUT_PATH = "data/processed/rul_labeled.csv"

# ----------------------------
# LOAD FILES
# ----------------------------
df = pd.read_csv(DATA_PATH)
fail_log = pd.read_csv(FAILURE_LOG)

print("✅ Sensor columns:", df.columns.tolist())
print("✅ Failure log columns:", fail_log.columns.tolist())

# ----------------------------
# TIMESTAMP DETECTION
# ----------------------------
if "time_stamp" in df.columns:
    ts_col = "time_stamp"
elif "timestamp" in df.columns:
    df = df.rename(columns={"timestamp": "time_stamp"})
    ts_col = "time_stamp"
else:
    raise ValueError("❌ No timestamp column found in sensor data")

df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

if "time_stamp" in fail_log.columns:
    f_ts_col = "time_stamp"
elif "timestamp" in fail_log.columns:
    fail_log = fail_log.rename(columns={"timestamp": "time_stamp"})
    f_ts_col = "time_stamp"
else:
    raise ValueError("❌ No timestamp column found in failure log")

fail_log[f_ts_col] = pd.to_datetime(fail_log[f_ts_col], errors="coerce")

# ----------------------------
# REMOVE DUPLICATE ID COLUMN (SAFE CLEAN)
# ----------------------------
if "asset_id" in df.columns and "id" in df.columns:
    if df["asset_id"].nunique() == 1 and df["id"].nunique() == 1:
        print("✅ Dropping redundant 'id' column")
        df = df.drop(columns=["id"])

# ----------------------------
# NORMALIZE 'failed' COLUMN
# ----------------------------
if "failed" not in fail_log.columns:
    raise ValueError("❌ failure_log.csv must contain a 'failed' column")

fail_log["failed"] = fail_log["failed"].astype(str).str.lower()
fail_log["failed"] = fail_log["failed"].map({
    "1": 1, "true": 1, "yes": 1,
    "0": 0, "false": 0, "no": 0
})

fail_log = fail_log.dropna(subset=["failed"])
fail_log["failed"] = fail_log["failed"].astype(int)

# ----------------------------
# SINGLE-TURBINE MODE (FOR YOUR DATASET)
# ----------------------------
print("✅ SINGLE-TURBINE MODE ENABLED")

fdf = fail_log[fail_log["failed"] == 1]

if len(fdf) == 0:
    raise ValueError("❌ No failure rows with failed=1 found in failure_log.csv")

# Take earliest real failure
t_fail = fdf.sort_values(f_ts_col).iloc[0][f_ts_col]

# ----------------------------
# CREATE RUL (HOURS)
# ----------------------------
df["RUL"] = (t_fail - df[ts_col]).dt.total_seconds() / 3600
df["RUL"] = df["RUL"].clip(lower=0)

# ----------------------------
# FINAL CLEANING & SAVE
# ----------------------------
df = df.dropna(subset=["RUL"])

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print("✅ RUL labels created successfully!")
print("📁 Saved to:", OUT_PATH)
print("✅ Total labeled samples:", len(df))
print("✅ Failure timestamp used:", t_fail)
print("✅ Sensor time range:",
      df[ts_col].min(), "to", df[ts_col].max())
