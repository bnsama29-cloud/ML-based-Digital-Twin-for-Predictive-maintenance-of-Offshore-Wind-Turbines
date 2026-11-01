import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -------------------------------
# PATHS
# -------------------------------
DATA_PATH = "data/processed/44_processed.csv"
MODEL_PATH = "models/rul_lstm_model.h5"
SCALER_PATH = "models/rul_scaler.pkl"
OUT_PATH = "data/processed/rul_predictions.csv"

SEQUENCE_LENGTH = 30

# -------------------------------
# LOAD MODEL & SCALER
# -------------------------------
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_PATH)

# -------------------------------
# DROP NON-NUMERICAL META COLUMNS
# -------------------------------
META_COLS = ["timestamp", "time_stamp", "asset_id", "id"]

SENSOR_COLS = [c for c in df.columns if c not in META_COLS]

df_sensors = df[SENSOR_COLS]

# -------------------------------
# SCALE (MATCH TRAINING)
# -------------------------------
X_scaled = scaler.transform(df_sensors.values)

# -------------------------------
# BUILD SEQUENCES
# -------------------------------
X_seq = []

for i in range(len(X_scaled) - SEQUENCE_LENGTH):
    X_seq.append(X_scaled[i:i + SEQUENCE_LENGTH])

X_seq = np.array(X_seq)

# -------------------------------
# PREDICT RUL
# -------------------------------
rul_preds = model.predict(X_seq, verbose=1).flatten()

# -------------------------------
# ALIGN PREDICTIONS TO TIMESTAMPS
# -------------------------------
df_out = df.iloc[SEQUENCE_LENGTH:].copy()
df_out["Predicted_RUL"] = rul_preds

# -------------------------------
# SAVE OUTPUT
# -------------------------------
df_out.to_csv(OUT_PATH, index=False)

print("✅ RUL prediction completed successfully!")
print("📁 Saved to:", OUT_PATH)
print("✅ Total predictions:", len(df_out))
