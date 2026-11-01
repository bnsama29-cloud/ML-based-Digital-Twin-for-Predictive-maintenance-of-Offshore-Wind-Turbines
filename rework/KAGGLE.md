# Running the rework on Kaggle (one account, one run)

One notebook does everything: all 95 CARE-to-Compare datasets (farms A, B, C:
45 faults, 50 normal periods), the dataset-44 deep dive, and the paper tables.

**Time:** about 2–3 h (estimate). The first run (farm C only, v1) took 73 min on Kaggle;
v2 adds farms A and B (smaller files) and one extra autoencoder per dataset. Kaggle's
limit is 12 h per run, so it fits.

## Notebook settings
- Settings → Internet **on** (streams files from Zenodo; needs a phone-verified account)
- Accelerator: GPU T4 if available (faster), CPU also works
- Click **Save Version → Save & Run All (Commit)** so it keeps running after you close the tab

## Cell 1 — setup
```python
!pip -q install remotezip
!git clone -q --depth 1 -b rework-2026 https://github.com/bnsama29-cloud/ml-based-digital-twin-for-predictive-maintenance-of-offshore-wind-turbines.git repo
%cd repo/rework
```

## Cell 2 — run everything
```python
!python run_all.py --data-dir /tmp/care --out /kaggle/working/results
```
If the run stops halfway, run the notebook again: finished datasets are skipped.

## What comes out (Output tab → download `results/`)
- `results/paper_tables/care.csv` / `care.tex` → CARE score table (all farms and per farm)
- `results/paper_tables/attribution.csv` → attribution table (all farms and per farm)
- `results/paper_tables/per_event.csv` → every dataset x detector x post-processing
- `results/deep44/` → case-study figures, `gate_sensitivity.csv`, `temperature_model.csv`,
  `robustness.csv`, `unity_telemetry.csv` (feed this to the Unity scene)
- `results/benchmark/scores/` → prediction-window scores of every row (any metric can be
  recomputed from these without retraining)

## Rules fixed before looking at results (do not change after seeing them)
- training rows: `train_test == train`, status 0/2, minus idle-in-good-wind rows
  (wind 5–20 m/s, power < 2 % of P99); last 15 % held out, split in time into C_a / C_b
- condition model: HistGBR on wind, power, rotor speed, ambient, fitted on C_a
- smoothing: causal 6 h rolling median
- point threshold: 99th percentile of the score on C_b
- alarm: CARE criticality counter ≥ 72 (12 h)
- AE: d-256-64-16-64-256-d, ReLU, linear output, Adam 1e-3, batch 256, ≤40 epochs,
  early stop patience 5, seeds 0/1/2; one extra AE without the training filter
