import pandas as pd
import numpy as np
import json
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# ---- LOAD DATA ----
df = pd.read_csv("data/processed/44_processed.csv", index_col=0)

# ---- NORMALIZE ----
X = StandardScaler().fit_transform(df)

# ---- CORRELATION DISTANCE ----
corr = pd.DataFrame(X, columns=df.columns).corr().fillna(0)
distance = 1 - np.abs(corr.values)

# ---- CLUSTER INTO 10 PHYSICAL SUBSYSTEMS ----
cluster = AgglomerativeClustering(
    n_clusters=10,
    metric="precomputed",
    linkage="average"
)
labels = cluster.fit_predict(distance)

# ---- AUTO ASSIGN GENERIC SUBSYSTEM TAGS ----
subsystems = [
    "ENVIRONMENT", "ROTOR", "SHAFT",
    "GEARBOX", "GENERATOR", "POWER_ELECTRONICS",
    "YAW", "PITCH", "TOWER", "GRID"
]

mapping = {
    sensor: subsystems[label]
    for sensor, label in zip(df.columns, labels)
}

# ---- SAVE JSON & CSV ----
with open("data/sensor_cluster_map.json", "w") as f:
    json.dump(mapping, f, indent=2)

pd.DataFrame({
    "sensor": df.columns,
    "subsystem": labels
}).to_csv("data/sensor_clusters.csv", index=False)

print("✅ Physical subsystem mapping created.")
