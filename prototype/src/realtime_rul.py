import pandas as pd
import numpy as np
import os

# ============================================================
# ✅ ✅ ✅ FINAL ENGINEERED PARAMETERS (TUNED FOR YOUR DATA)
# ============================================================
ROLL_WIN = 40              # window for local linear slope estimation
SMOOTH_SPAN = 50           # EWMA smoothing for health
MIN_SLOPE = 0.002          # minimum meaningful degradation rate (health/hour)
MAX_RUL = 600.0            # realistic offshore turbine prediction horizon (hours)
FAILURE_HEALTH = 0.05      # failure threshold
MEDIAN_SMOOTH_RUL = 5      # median smoothing window on RUL

HEALTH_PATH = "data/processed/health_index.csv"
OUT_PATH = "data/processed/realtime_rul.csv"

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================
df = pd.read_csv(HEALTH_PATH)

if "time_stamp" in df.columns:
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
elif "timestamp" in df.columns:
    df = df.rename(columns={"timestamp": "time_stamp"})
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
else:
    raise ValueError("❌ No timestamp column found in health_index.csv")

df = df.sort_values("time_stamp").reset_index(drop=True)

if "health_index" not in df.columns:
    raise ValueError("❌ health_index column not found in health_index.csv")

health = df["health_index"].astype(float)

# ============================================================
# 2. SMOOTH HEALTH INDEX
# ============================================================
health_smooth = health.ewm(span=SMOOTH_SPAN, adjust=False).mean()

# Time in hours
time_hours = (df["time_stamp"] - df["time_stamp"].iloc[0]).dt.total_seconds() / 3600
dt_median = np.median(np.diff(time_hours))
if np.isnan(dt_median) or dt_median <= 0:
    dt_median = 1.0  # safe fallback

# ============================================================
# 3. ROBUST ROLLING LINEAR SLOPE (PER SAMPLE → PER HOUR)
# ============================================================
def rolling_slope(series, win):
    arr = series.values
    n = len(arr)
    slopes = np.full(n, np.nan)
    half = win // 2

    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        seg = arr[i0:i1]

        if len(seg) < max(6, win // 2):
            continue

        x = np.arange(len(seg))
        x_mean = x.mean()
        y_mean = seg.mean()

        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            continue

        slope_local = ((x - x_mean) * (seg - y_mean)).sum() / denom
        slopes[i] = slope_local

    return slopes


raw_slope_per_sample = rolling_slope(health_smooth, ROLL_WIN)
slope_per_hour = raw_slope_per_sample / dt_median

slope_series = pd.Series(slope_per_hour).fillna(0.0)

# Enforce physical degradation direction (health must not improve)
slope_series[slope_series > 0] = 0.0

# ============================================================
# 4. STABLE REAL-TIME RUL COMPUTATION (SEQUENTIAL)
# ============================================================
rul_list = []
prev_rul = None

for i in range(len(df)):
    h = float(health_smooth.iloc[i])
    s = float(slope_series.iloc[i])

    # If slope is too small → no measurable degradation
    if abs(s) < MIN_SLOPE:
        if prev_rul is None:
            estimated = (h - FAILURE_HEALTH) / (MIN_SLOPE + 1e-9)
            estimated = np.clip(estimated, 1.0, MAX_RUL)
            rul_i = estimated
        else:
            # ✅ Apply slow time-based decay instead of freezing
            rul_i = prev_rul - dt_median * 0.5   # 0.5 hour decay per timestep
    else:
        raw_rul = (h - FAILURE_HEALTH) / (abs(s) + 1e-9)
        raw_rul = np.clip(raw_rul, 0.0, MAX_RUL)
        rul_i = raw_rul

    # Enforce monotonic non-increasing RUL
    if prev_rul is not None:
        rul_i = min(prev_rul, rul_i)

    rul_list.append(rul_i)
    prev_rul = rul_i

rul_series = pd.Series(rul_list)

# ============================================================
# 5. FINAL SMOOTHING & SAFETY CLIPS
# ============================================================
rul_series = rul_series.rolling(
    MEDIAN_SMOOTH_RUL, min_periods=1, center=True
).median()

rul_series = rul_series.clip(0.0, MAX_RUL)
rul_series = rul_series.fillna(method="ffill").fillna(MAX_RUL)

# ============================================================
# 6. SAVE OUTPUT
# ============================================================
out = pd.DataFrame({
    "timestamp": df["time_stamp"],
    "health_index": health_smooth,
    "health_slope_per_hour": slope_series,
    "RealTime_RUL_hours": rul_series
})

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out.to_csv(OUT_PATH, index=False)

print("✅ Robust Real-Time RUL generated")
print("📁 Saved to:", OUT_PATH)
print("✅ RUL range (hours):", rul_series.min(), "to", rul_series.max())
