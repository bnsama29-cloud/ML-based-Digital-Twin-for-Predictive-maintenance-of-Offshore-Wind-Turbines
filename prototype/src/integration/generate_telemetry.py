import pandas as pd
from datetime import datetime
import random
import os

OUTPUT_PATH = "data/processed/latest_telemetry.csv"

def generate_fake_telemetry():
    telemetry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rpm": round(random.uniform(1200, 1650), 2),
        "wind_speed": round(random.uniform(5, 15), 2),
        "power_output": round(random.uniform(1.5, 4.0), 2),
        "temperature": round(random.uniform(50, 80), 2),
        "status": random.choice(["OK", "WARN", "FAULT"])
    }

    df = pd.DataFrame([telemetry])
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Telemetry updated:", telemetry)

if __name__ == "__main__":
    generate_fake_telemetry()
