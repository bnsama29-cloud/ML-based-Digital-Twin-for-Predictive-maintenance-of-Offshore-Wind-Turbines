import json
from typing import List


def load_sensor_cluster_map(json_path: str):
    """Load sensor → subsystem mapping."""
    with open(json_path, "r") as f:
        return json.load(f)


def map_sensors_to_subsystems(
    sensor_list: List[str],
    sensor_map: dict
):
    """
    Converts:
        ['sensor_12', 'sensor_87']
    Into:
        ['GEARBOX', 'GENERATOR']
    """
    subsystems = []

    for s in sensor_list:
        subsystem = sensor_map.get(s, "UNKNOWN")
        subsystems.append(subsystem)

    # Remove duplicates while preserving order
    subsystems = list(dict.fromkeys(subsystems))
    return subsystems


def format_rca_output(sensor_list: List[str], sensor_map: dict):
    subsystems = map_sensors_to_subsystems(sensor_list, sensor_map)
    return " + ".join(subsystems)
