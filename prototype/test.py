import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/health_index.csv")
df['time_stamp'] = pd.to_datetime(df['time_stamp'])
df = df.sort_values('time_stamp')

# smoothed health (same as in pipeline)
health_sm = df['health_index'].astype(float).ewm(span=30).mean()
dt_hours = (df['time_stamp'].diff().dt.total_seconds().median()) / 3600.0

# approx slope (per sample)
slope = np.gradient(health_sm, dt_hours)
print("slope stats:", pd.Series(slope).describe())

# count tiny slopes
eps = 1e-4
print("tiny slope count:", (np.abs(slope) < eps).sum())

# compute raw_rul before clipping for inspection
failure_h = 0.05
raw_rul = (health_sm - failure_h) / (np.abs(slope) + 1e-12)
print("raw_rul stats:", raw_rul.describe())
print("count > 2000:", (raw_rul > 2000).sum())
