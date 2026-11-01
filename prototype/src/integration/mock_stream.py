import time
import pandas as pd
import random

while True:
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp.now()],
        "rotor_speed": [random.uniform(8, 16)],
        "power_output": [random.uniform(100, 1900)],
        "vibration_level": [random.uniform(0.1, 2.1)]
    })

    df.to_csv("data/processed/latest_telemetry.csv", index=False)
    time.sleep(1)
