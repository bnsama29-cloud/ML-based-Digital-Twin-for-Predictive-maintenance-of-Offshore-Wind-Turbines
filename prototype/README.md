# First prototype (semester project, 2025)

This folder holds the first version of the project: a TensorFlow autoencoder on one
CARE-to-Compare turbine (dataset 44), a health index, an LSTM remaining-useful-life model,
a Streamlit dashboard, a FastAPI/MQTT bridge and a telemetry generator for the Unity scene.
It is kept for the record. **It is not used for any reported result**; the validated
pipeline is in [`../pipeline`](../pipeline).

## Why it was replaced
Review of this version found problems that the new pipeline fixes:

| Problem in the prototype | Fixed in `pipeline/` by |
|---|---|
| Trained and scored on one file that already contained the fault window | training on the normal-status training year only; prediction window never seen before scoring |
| Row id, asset id and status code fed to the model as sensors | identifiers and status removed from the inputs |
| Centred rolling median on the RUL output (reads future rows); health index forced to only decrease (`np.minimum.accumulate`); imputer and outlier clipping fitted on the whole file | causal windows everywhere; every statistic fitted on training rows only; health index recomputed per row |
| No detection metrics, baselines or repeated runs | CARE score on all 95 datasets, five baselines, three seeds, paired bootstrap |
| "Remaining useful life" with no failure dates to validate against | time-to-threshold with a 95 % band, reported as a planning aid |
| Unity telemetry (`src/integration/generate_unity_telemetry.py`) synthesised wind and rotor speed from the health index | the case-study export (`pipeline/analyze_case.py`) writes measured SCADA values only |

## Contents
| Path | What it is |
|---|---|
| `colab_prototype.ipynb` | original Colab notebook |
| `src/preprocess.py`, `src/train.py`, `src/infer.py` | preprocessing, autoencoder training and scoring for dataset 44 |
| `src/build_health_index.py`, `src/rca_subsystem_mapper.py` | health index and sensor-group attribution |
| `src/prepare_rul_labels.py`, `src/rul_train.py`, `src/rul_infer.py`, `src/realtime_rul.py` | LSTM RUL model and its online version |
| `src/predictive_maintenance.py`, `src/generate_health_report.py` | maintenance schedule and HTML report |
| `src/integration/` | FastAPI twin API, MQTT publisher and telemetry generators for Unity |
| `dashboard.py` | Streamlit dashboard |
| `scripts/` | sensor-to-subsystem map builders |
| `models/`, `data/processed/` | trained weights and generated reports from the prototype runs |

Install with `pip install -r prototype/requirements.txt` and run the scripts from inside
this folder; paths are relative to it.
