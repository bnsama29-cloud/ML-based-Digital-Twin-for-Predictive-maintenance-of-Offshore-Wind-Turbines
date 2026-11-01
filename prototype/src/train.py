import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import matplotlib.pyplot as plt
import os

PROCESSED_PATH = "data/processed/44_processed.csv"
MODEL_PATH = "models/autoencoder.h5"


def main():
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_PATH, index_col=0)
    X = df.values

    # Train / validation split (time-based)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]

    input_dim = X.shape[1]
    latent_dim = max(4, input_dim // 4)

    print("Building autoencoder...")
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(latent)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = models.Model(inputs, outputs)
    autoencoder.compile(optimizer="adam", loss="mse")

    print("Training...")
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=25,
        batch_size=256,
        shuffle=True
    )

    os.makedirs("models", exist_ok=True)
    autoencoder.save(MODEL_PATH)

    print("✅ Model saved to", MODEL_PATH)

    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
