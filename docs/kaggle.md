# Running the full experiment on Kaggle

The whole study is two notebook runs: **run A** (all detectors, case studies, result tables)
and **run B** (LSTM autoencoder and compute-time measurements). Run B is optional and can run
in parallel on a second person's account. Each run downloads only the files it needs from
Zenodo and ends with a zip in the Output tab.

| | Run A | Run B |
|---|---|---|
| Detectors | AE ×3 seeds, AE without idle filter, conditional AE ×3, sigmoid-output AE, Isolation Forest ×3, PCA, 3σ | LSTM AE ×3 seeds |
| Extra | robustness tests, two case studies | fit/score time of every detector |
| Measured time (T4 GPU) | 5.3 h | 4.8 h |
| Output | `results_v3.zip`, `results_v3_scores.zip` | `results_v3_lstm.zip`, `results_v3_lstm_scores.zip` |

A 10-hour guard stops new datasets from starting late, so the tables and the zip are always
written before Kaggle's 12-hour limit.

## Notebook settings
- Settings → **Internet on** (phone-verified account)
- Accelerator → **GPU T4**
- **Save Version → Save & Run All (Commit)**, so the run continues after the tab is closed

## Cell 1: setup (clones to /tmp so the repo stays out of the Output tab)
```python
!pip -q install remotezip
!git clone -q --depth 1 https://github.com/bnsama29-cloud/ML-based-Digital-Twin-for-Predictive-maintenance-of-Offshore-Wind-Turbines.git /tmp/repo
%cd /tmp/repo
!python -m pytest -q tests
```
The tests must report `39 passed`.

## Cell 2: run
Run A:
```python
!python pipeline/run_all.py --data-dir /tmp/care --out /tmp/results --zip /kaggle/working/results_v3.zip
```
Run B:
```python
!python pipeline/run_all.py --detectors lstm --data-dir /tmp/care --out /tmp/results_b --zip /kaggle/working/results_v3_lstm.zip
```

## Cell 3: download links
```python
from IPython.display import FileLink, display
for f in ("results_v3.zip", "results_v3_scores.zip"):          # run B: results_v3_lstm*.zip
    display(FileLink(f))
```

## Merging run A and run B
Unzip both score archives, then:
```bash
python pipeline/merge_results.py --inputs A/results_v3/benchmark B/results_v3/benchmark --out results/tables
python pipeline/make_figures.py --tables results/tables --out figures
```
Datasets are matched by farm and event id, and the script checks that both runs saw the
same rows.

## What the outputs contain
| File | Content |
|---|---|
| `tables/care.csv` | CARE score and its parts for every detector × post-processing, all farms and per farm |
| `tables/bootstrap.csv` | paired, label-stratified bootstrap of CARE differences between configurations |
| `tables/lofo.csv` | leave-one-farm-out choice of quantile and counter, plus the in-sample ceiling |
| `tables/counter_sensitivity.csv` | alarm counter of 36 / 72 / 144 rows |
| `tables/robustness.csv` | CARE, detections and false alarms under noise and missing values |
| `tables/val_loss_heads.csv` | best validation loss: linear vs sigmoid output, conditional AE, LSTM AE |
| `tables/attribution.csv` | plant-area attribution vs random, majority-class and z-score floors |
| `tables/timing.csv` | fit and scoring time per detector and farm |
| `tables/per_event.csv` | every dataset × detector × seed × post-processing |
| `case_44/`, `case_40/` | health index, time-to-threshold, gate sensitivity, attribution, figures; temperature model and Unity CSV for farm C |
| `RUN_INFO.txt` | total time and any failed step |

## Settings fixed before the runs
- Training rows: training year, status 0/2, minus idle-in-good-wind rows (wind 5–20 m/s,
  power < 2 % of P99); the final 15 % of these rows form C_a (first half) and C_b (second half)
- Condition model: HistGBR (200 trees, rate 0.05) on wind, power, rotor speed, ambient, fitted on C_a
- Smoothing: causal 6 h rolling median
- Threshold: 99th percentile on C_b; alarm: criticality counter ≥ 72 (12 h)
- AE: d-256-64-16-64-256-d, ReLU, linear output, Adam 1e-3, batch 256, ≤ 40 epochs, patience 5
- Conditional AE: same widths, the four scaled operating conditions added to input and code
- LSTM AE: 1 h windows (6 rows), hidden size 64, scored causally
- Tuning grid (leave-one-farm-out only): quantile {0.95, 0.975, 0.99, 0.995, 0.999} × counter {36, 72, 144}
- Robustness: only the test window is altered (Gaussian noise σ ∈ {0.1, 0.3, 0.5}, or 5 % /
  20 % of values replaced by the training median); thresholds stay as calibrated
- Case studies: dataset 44 (cooling-water valve, farm C) and dataset 40 (generator bearing, farm A),
  both chosen before the run
