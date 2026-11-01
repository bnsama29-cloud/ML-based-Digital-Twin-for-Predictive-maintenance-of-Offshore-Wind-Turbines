# Fault detection, attribution and maintenance timing on wind-turbine SCADA data

Code for the paper *Condition-normalised autoencoder scoring for fault detection,
attribution and maintenance timing on wind turbine SCADA data* (RV College of Engineering).

The study is a leakage-safe evaluation on **all 95 datasets of the
[CARE-to-Compare](https://zenodo.org/records/15846963) benchmark** (3 farms, 45 faults,
50 normal periods), scored with the benchmark's CARE score. It covers:

- detectors: dense autoencoder, conditional autoencoder, Isolation Forest, PCA, 3σ fraction
- post-processing: condition normalisation (remove the error that wind, power, rotor speed
  and ambient temperature explain), causal smoothing, both, ablated one by one
- plant-area attribution checked against operator fault notes, with random, majority-class
  and no-model floors
- a causal health index with a time-to-threshold band, a gate sensitivity study, and a
  Unity playback that only shows measured values and model outputs

All settings were fixed before testing (see [`rework/KAGGLE.md`](rework/KAGGLE.md)).

## Results so far (v2 run)

| Detector | Post-processing | CARE (3 seeds) | Faults detected / 45 | Normal periods with alarm / 50 |
|---|---|---|---|---|
| Autoencoder | raw | 0.584 ± 0.002 | 11.3 | 5.0 |
| Autoencoder | + smoothing | 0.595 ± 0.004 | 17.7 | 8.3 |
| Autoencoder | + cond. norm. | 0.568 ± 0.003 | 10.3 | 4.0 |
| Autoencoder | + both | 0.592 ± 0.004 | 15.3 | 7.0 |
| PCA | + both | 0.598 | 16 | 9 |
| Isolation Forest | raw → + both | 0.467 → 0.515 | 2 → 9 | 5 → 9 |

Reference values from the benchmark paper: tuned autoencoder 0.66, random 0.5.

A paired bootstrap over datasets finds no significant difference between the autoencoder
variants or between the autoencoder and PCA. The only significant gain is condition
normalisation plus smoothing on Isolation Forest: +0.048, 95 % CI [0.003, 0.094].
Attribution top-1 is 0.36, against 0.12 for a random ranking and 0.30 for always naming
the commonest area.

Tables are in [`rework/results/v2/`](rework/results/v2). The v3 run adds the conditional AE,
leave-one-farm-out threshold tuning, robustness on all datasets, a sigmoid-head ablation
and a second case study.

## Layout

```
rework/            the study (use this)
  wtpm_lib.py        data access, preprocessing, models, CARE score, attribution
  run_benchmark.py   all 95 datasets -> stored scores
  analyze_case.py    one-dataset case study (health index, time to threshold, gate, RCA)
  merge_results.py   stored scores -> paper tables
  run_all.py         everything in one go, ends with a zip
  tests/             pytest suite
  paper/main.tex     IEEE conference paper
  results/v2/        tables of the v2 run
src/, dashboard.py   first prototype (kept for history; not used by the paper)
```

## Run it

On Kaggle: follow [`rework/KAGGLE.md`](rework/KAGGLE.md). It takes one notebook and one
run, and the output is a single zip.

Locally (downloads only the files it needs from Zenodo, by HTTP range requests):

```bash
cd rework
pip install -r requirements.txt
python -m pytest -q tests
python run_all.py --data-dir data --out out --zip out/results_v3.zip --events 7 21 40 --epochs 2   # smoke test
python run_all.py --data-dir data --out out --zip out/results_v3.zip                               # full run
```

## Data and licence of the data

CARE-to-Compare by Gück, Roelofs and Faulstich (Fraunhofer IEE), *Data* 9(12):138, 2024,
distributed via Zenodo under its own licence. The data are not copied into this repository.

## Authors

N Ananth, Roshan Kalirajan, Samarth B N, Veena Gadad (RV College of Engineering, Bengaluru).
