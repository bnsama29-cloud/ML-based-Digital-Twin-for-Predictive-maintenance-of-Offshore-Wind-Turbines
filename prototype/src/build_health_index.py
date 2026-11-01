import pandas as pd
import numpy as np
import os

ANOM_PATH = "data/processed/anomaly_with_root_cause.csv"
OUT_PATH = "data/processed/health_index.csv"

df = pd.read_csv(ANOM_PATH)

# Timestamp normalize
if "time_stamp" in df.columns:
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
elif "timestamp" in df.columns:
    df = df.rename(columns={"timestamp": "time_stamp"})
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
else:
    raise ValueError("❌ No timestamp column found")

# Select anomaly intensity
if "anomaly_score" in df.columns:
    raw = df["anomaly_score"].astype(float)
elif "reconstruction_error" in df.columns:
    raw = df["reconstruction_error"].astype(float)
elif "is_anomaly" in df.columns:
    raw = df["is_anomaly"].astype(float)
else:
    raise ValueError("❌ No anomaly intensity column found")

# Smooth
raw_smooth = raw.ewm(span=60).mean()

# ✅ ROLLING BASELINE NORMALIZATION (KEY FIX)
rolling_min = raw_smooth.rolling(500, min_periods=50).min()
rolling_max = raw_smooth.rolling(500, min_periods=50).max()

norm = (raw_smooth - rolling_min) / (rolling_max - rolling_min + 1e-6)
norm = norm.clip(0, 1)

# Health index
health = 1.0 - norm
health = np.minimum.accumulate(health.fillna(method="bfill"))
health = health.clip(0.05, 1.0)

df_out = pd.DataFrame({
    "time_stamp": df["time_stamp"],
    "health_index": health
})

os.makedirs("data/processed", exist_ok=True)
df_out.to_csv(OUT_PATH, index=False)

print("✅ Robust Health Index generated")
print("📁 Saved to:", OUT_PATH)
print("✅ Health range:", df_out["health_index"].min(), "to", df_out["health_index"].max())
