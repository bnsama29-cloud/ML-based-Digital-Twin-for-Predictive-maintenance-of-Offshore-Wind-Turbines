import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os

RAW_PATH = "data/raw/44.csv"
PROCESSED_PATH = "data/processed/44_processed.csv"
SCALER_PATH = "models/scaler.joblib"
IMPUTER_PATH = "models/imputer.joblib"


def main():
    print("Loading data...")
    df = pd.read_csv(RAW_PATH, sep=";", engine="python")
    print(f"Loaded shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Parse timestamp (assumes first column is datetime)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df = df.dropna(subset=[df.columns[0]])
    print(f"After timestamp parsing: {df.shape}")
    df = df.set_index(df.columns[0]).sort_index()

    # Convert all columns to numeric if possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only numeric sensor columns
    sensor_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
    print(f"Numeric columns found: {len(sensor_cols)}")
    
    if len(sensor_cols) == 0:
        raise ValueError("No numeric columns found in data!")
    
    df = df[sensor_cols]
    print(f"After column filtering: {df.shape}")

    
    # Drop sensors with >30% missing values
    keep = df.isna().mean() <= 0.30
    df = df.loc[:, keep]

    # Interpolate short gaps
    df = df.interpolate(limit=5)

    # Impute remaining NaNs with median
    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df),
        index=df.index,
        columns=df.columns
    )

    # Clip outliers
    q_low = df_imputed.quantile(0.01)
    q_high = df_imputed.quantile(0.99)
    df_clipped = df_imputed.clip(q_low, q_high, axis=1)

    # Scale
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_clipped),
        index=df_clipped.index,
        columns=df_clipped.columns
    )

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    df_scaled.to_csv(PROCESSED_PATH)

    print("✅ Preprocessing complete")
    print("Saved:", PROCESSED_PATH)


if __name__ == "__main__":
    main()
