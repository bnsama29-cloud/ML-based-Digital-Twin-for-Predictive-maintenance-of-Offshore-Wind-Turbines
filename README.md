# A leakage-safe autoencoder framework for early fault detection and subsystem attribution in wind turbines

Code and results for the paper *A Leakage-Safe Autoencoder Framework for Early Fault
Detection and Subsystem Attribution in Wind Turbines Using SCADA Data* (RV College of
Engineering, Bengaluru).

The study evaluates fault detectors on **all 95 datasets of the
[CARE-to-Compare](https://zenodo.org/records/15846963) benchmark** (3 farms, 45 faults,
50 normal periods) with time-ordered splits, settings fixed before testing, three seeds,
and the benchmark's CARE score. It covers:

- detectors: dense autoencoder (AE), conditional AE, LSTM AE, AE with a sigmoid output,
  PCA, Isolation Forest, 3σ fraction
- post-processing: condition normalisation (remove the error that wind, power, rotor speed
  and ambient temperature explain), causal smoothing, both — each ablated
- leave-one-farm-out threshold tuning, a paired bootstrap over datasets, robustness to
  noise and missing data, compute time
- plant-area attribution checked against operator fault notes, with random, majority-class
  and model-free floors
- a causal health index with a time-to-threshold band, a work-order gate study, and a
  Unity playback of measured values

## Results (final run, v3)

| Detector | Post-processing | CARE, fixed settings | CARE, tuned leave-one-farm-out | Faults detected / 45 | Normal periods with alarm / 50 |
|---|---|---|---|---|---|
| AE | raw | 0.584 ± 0.002 | 0.607 | 11.3 | 5.0 |
| AE | + normalisation + smoothing | 0.592 ± 0.004 | 0.602 | 15.3 | 7.0 |
| Conditional AE | raw | 0.590 ± 0.007 | 0.603 | 11.7 | 3.7 |
| LSTM AE | raw | 0.531 ± 0.000 | 0.601 | 5.0 | 3.0 |
| AE, sigmoid output | raw | 0.498 | – | 6 | 10 |
| PCA | + normalisation + smoothing | 0.598 | 0.611 | 16 | 9 |
| Isolation Forest | raw → + both | 0.471 → 0.518 | 0.527 → 0.582 | 2.3 → 9.3 | 4.3 → 8.3 |

Reference values from the benchmark paper: per-farm tuned AE 0.66, random 0.5.

- A paired bootstrap over datasets finds no significant difference between the AE
  post-processing variants, or between AE, conditional AE and PCA. The LSTM AE is lower
  (−0.054, 95 % CI −0.092 to −0.019); the sigmoid output costs 0.086 (0.025 to 0.146).
  Normalisation + smoothing helps Isolation Forest (+0.047, 0.003 to 0.091).
- Tuning thresholds on two farms and testing on the third adds 0–0.07; even tuning on
  the test data gives at most 0.619, so most of the gap to 0.66 is not threshold choice.
- Attribution top-1: 0.37 (AE vote) vs 0.22 (raw z-scores), 0.12 (random), 0.30 (always the
  commonest area).
- Case studies: generator-bearing fault (dataset 40) detected ~29 days before the end of
  its window with the generator ranked first; cooling-valve fault (dataset 44) missed,
  cooling still ranked first.

All tables: [`rework/results/v3/`](rework/results/v3) (v2 kept for reference).

## Layout

```
rework/            the study (use this)
  wtpm_lib.py        data access, preprocessing, models, CARE score, attribution
  run_benchmark.py   all 95 datasets -> stored scores
  analyze_case.py    one-dataset case study (health index, time to threshold, gate, RCA)
  merge_results.py   stored scores -> paper tables
  run_all.py         everything in one go, ends with a zip
  tests/             pytest suite
  paper/main.tex     IEEE conference paper (make_figures.py draws Fig. 1 and 2)
  results/v3/        tables of the final run (results/v2: earlier run)
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
python run_all.py --data-dir data --out out --zip out/results_v3.zip                               # run A
python run_all.py --detectors lstm --data-dir data --out out_b --zip out/results_v3_lstm.zip       # run B
python merge_results.py --inputs out/benchmark out_b/benchmark --out tables                      # merge
```

## Data and licence of the data

CARE-to-Compare by Gück, Roelofs and Faulstich (Fraunhofer IEE), *Data* 9(12):138, 2024,
distributed via Zenodo under its own licence. The data are not copied into this repository.

## Authors

N Ananth, Roshan Kalirajan, Samarth B N, Veena Gadad (RV College of Engineering, Bengaluru).
