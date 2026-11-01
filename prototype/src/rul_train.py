import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = "data/processed/rul_labeled.csv"
MODEL_OUT = "models/rul_lstm_model.h5"

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_PATH)

# -------------------------------
# REMOVE META COLUMNS SAFELY
# -------------------------------
META_COLS = ["timestamp", "time_stamp", "asset_id", "id", "RUL"]

SENSOR_COLS = [c for c in df.columns if c not in META_COLS]

X_raw = df[SENSOR_COLS]
y = df["RUL"].values

# -------------------------------
# SCALE FEATURES (VECTOR SAFE)
# -------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# Save scaler
os.makedirs("models", exist_ok=True)
pd.to_pickle(scaler, "models/rul_scaler.pkl")

# -------------------------------
# SEQUENCE GENERATION (LSTM)
# -------------------------------
SEQUENCE_LENGTH = 30

X_seq = []
y_seq = []

for i in range(len(X_scaled) - SEQUENCE_LENGTH):
    X_seq.append(X_scaled[i:i+SEQUENCE_LENGTH])
    y_seq.append(y[i+SEQUENCE_LENGTH])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# -------------------------------
# TRAIN / VALIDATION SPLIT
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

# -------------------------------
# BUILD LSTM MODEL
# -------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# -------------------------------
# TRAIN
# -------------------------------
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save(MODEL_OUT)

print("✅ RUL LSTM model trained successfully!")
print("📁 Model saved to:", MODEL_OUT)
print("✅ Training samples:", X_train.shape[0])
print("✅ Validation samples:", X_val.shape[0])
