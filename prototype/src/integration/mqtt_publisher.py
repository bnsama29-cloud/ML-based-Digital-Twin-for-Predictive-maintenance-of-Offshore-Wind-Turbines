import json
import time
import paho.mqtt.client as mqtt
import pandas as pd

client = mqtt.Client()
client.connect("localhost", 1883, 60)

while True:
    df = pd.read_csv("data/processed/rul_predictions.csv")
    client.publish(
        "turbine/rul",
        json.dumps(df.to_dict(orient="records"))
    )
    time.sleep(1)
