# Final run on Kaggle (v3, one account, one run)

One notebook does everything: all 95 CARE-to-Compare datasets (farms A, B, C:
45 faults, 50 normal periods), two case studies, every paper table, and a zip at the end.

**Time:** roughly twice the v2 run, since v3 trains 8 networks per dataset instead of 4
(an estimate, not measured). A 10 h guard stops starting new datasets in time, so the
tables and the zip are always written before Kaggle's 12 h limit.

## Notebook settings
- Settings → Internet **on** (streams files from Zenodo; needs a phone-verified account)
- Accelerator: **GPU T4** (P100 also works)
- **Save Version → Save & Run All (Commit)**, so it keeps running after you close the tab

## Cell 1: setup (the repo goes to /tmp, so it does not end up in the Output tab)
```python
!pip -q install remotezip
!git clone -q --depth 1 https://github.com/bnsama29-cloud/ML-based-Digital-Twin-for-Predictive-maintenance-of-Offshore-Wind-Turbines.git /tmp/repo
%cd /tmp/repo/rework
!python -m pytest -q tests
```

## Cell 2: run everything
```python
!python run_all.py --data-dir /tmp/care --out /tmp/results --zip /kaggle/working/results_v3.zip
```

## Cell 3: download link
```python
from IPython.display import FileLink, display
display(FileLink('results_v3.zip'))          # tables, figures, logs: this is the one to send
display(FileLink('results_v3_scores.zip'))   # raw scores, only needed to recompute metrics
```
The Output tab then holds just these two files.

## What is in results_v3.zip
| File | What it answers |
|---|---|
| `paper_tables/care.csv`, `care.tex` | CARE score of every detector x post-processing, all farms and per farm |
| `paper_tables/bootstrap.csv` | paired bootstrap over datasets: which differences are real |
| `paper_tables/lofo.csv` | thresholds tuned on two farms, tested on the third |
| `paper_tables/counter_sensitivity.csv` | alarm counter 36 / 72 / 144 |
| `paper_tables/robustness.csv` | CARE under noise and missing values, all datasets |
| `paper_tables/val_loss_heads.csv` | linear vs sigmoid output head (CSITSS reviewer 5) |
| `paper_tables/attribution.csv` | plant-area attribution vs floors and a no-model z-score ranker |
| `case_44/`, `case_40/` | case-study figures, gate sensitivity, temperature model, Unity CSV |
| `RUN_INFO.txt` | total time and any failed step |

## Rules fixed before the v3 run (do not change after seeing results)
- training rows: `train_test == train`, status 0/2, minus idle-in-good-wind rows
  (wind 5–20 m/s, power < 2 % of P99); last 15 % held out, split in time into C_a / C_b
- condition model: HistGBR on wind, power, rotor speed, ambient, fitted on C_a
- smoothing: causal 6 h rolling median
- main threshold: 99th percentile of the score on C_b; alarm: counter ≥ 72 (12 h)
- AE: d-256-64-16-64-256-d, ReLU, linear output, Adam 1e-3, batch 256, ≤ 40 epochs,
  patience 5, seeds 0/1/2; conditional AE: same widths, the four scaled operating
  conditions appended to encoder input and code; sigmoid head: seed 0 only
- tuning grid (leave-one-farm-out only): quantile {0.95, 0.975, 0.99, 0.995, 0.999} x
  counter {36, 72, 144}, chosen by seed-averaged CARE on the two other farms
- robustness: prediction window only, Gaussian noise sd {0.1, 0.3, 0.5} or {5, 20} % entries
  set to the training median; clean thresholds; AE seed 0 and PCA
- case studies: dataset 44 (cooling valve, missed in v2, kept as the honest miss) and
  dataset 40 (farm A generator bearing, chosen before v3 as the detected example)
