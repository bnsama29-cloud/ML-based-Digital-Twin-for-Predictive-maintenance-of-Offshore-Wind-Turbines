import pandas as pd
import numpy as np

# =============================
# CONFIG
# =============================
INPUT_CSV = "data/processed/realtime_rul.csv"
OUTPUT_CSV = "data/processed/telemetry_history.csv"

RATED_POWER_KW = 3000
CUT_IN_WIND = 3.0
RATED_WIND = 12.0
CUT_OUT_WIND = 25.0

np.random.seed(42)

# =============================
# LOAD BASE DATA
# =============================
df = pd.read_csv(INPUT_CSV)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)

df["health_index"] = df["health_index"].clip(0.05, 1.0)
df["RealTime_RUL_hours"] = df["RealTime_RUL_hours"].clip(lower=0)

n = len(df)

# =============================
# WIND SPEED (m/s)
# Smooth offshore-like variation
# =============================
time_factor = np.linspace(0, 4 * np.pi, n)
slow_variation = 1.5 * np.sin(time_factor)

df["wind_speed_ms"] = (
    6
    + 6 * df["health_index"]
    + slow_variation
)

df["wind_speed_ms"] = df["wind_speed_ms"].clip(3, 20)

# =============================
# ROTOR SPEED (RPM)
# =============================
def rotor_speed(wind):
    if wind < CUT_IN_WIND:
        return 0.5
    elif wind < RATED_WIND:
        return 6 + (wind - CUT_IN_WIND) / (RATED_WIND - CUT_IN_WIND) * 9
    else:
        return 15

df["rotor_speed_rpm"] = df["wind_speed_ms"].apply(rotor_speed)
df["rotor_speed_rpm"] *= df["health_index"]

# =============================
# POWER OUTPUT (kW)
# =============================
def power_output(wind):
    if wind < CUT_IN_WIND:
        return 0
    elif wind < RATED_WIND:
        return RATED_POWER_KW * (wind / RATED_WIND) ** 3
    elif wind < CUT_OUT_WIND:
        return RATED_POWER_KW
    else:
        return 0

df["power_output_kw"] = df["wind_speed_ms"].apply(power_output)
df["power_output_kw"] *= df["health_index"]

# =============================
# TEMPERATURES (°C)
# =============================
load_fraction = df["power_output_kw"] / RATED_POWER_KW

df["gearbox_temperature_c"] = (
    60
    + 25 * load_fraction
    + 15 * (1 - df["health_index"])
)

df["generator_temperature_c"] = (
    55
    + 20 * load_fraction
    + 12 * (1 - df["health_index"])
)

# =============================
# FINALIZE
# =============================
df = df.rename(columns={
    "RealTime_RUL_hours": "rul_hours"
})

final_cols = [
    "timestamp",
    "wind_speed_ms",
    "rotor_speed_rpm",
    "power_output_kw",
    "gearbox_temperature_c",
    "generator_temperature_c",
    "health_index",
    "rul_hours"
]

df = df[final_cols]

df.to_csv(OUTPUT_CSV, index=False)

print("✅ Realistic Unity telemetry generated")
print(f"📁 Saved to: {OUTPUT_CSV}")
print(df.head())
