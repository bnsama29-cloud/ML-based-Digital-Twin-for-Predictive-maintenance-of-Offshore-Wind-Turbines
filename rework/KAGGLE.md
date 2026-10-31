# Running the rework on Kaggle (one account, one run)

One notebook does everything: all 58 farm C datasets (26 anomaly, 32 normal), the
dataset-44 deep dive, and the paper tables.

**Time:** about 3–5 h in total (estimate). Measured locally: 2.3 min per dataset on a
2-core CPU plus a ~300 MB download each; the deep dive took 1.3 min. Kaggle's limit is
12 h per run, so it fits.

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
- `results/paper_tables/detection.tex`, `attribution.tex` → Tables II and III
- `results/deep44/` → Figs. 2–5, `gate_sensitivity.csv`, `temperature_model.csv`,
  `robustness.csv`, `unity_telemetry.csv` (feed this to the Unity scene)
- `results/benchmark/` → raw per-dataset results and training logs

## Rules fixed before looking at results (do not change after seeing them)
- training rows: `train_test == train` and status 0/2; last 15 % = validation
- point threshold: 99th percentile of validation score
- alarm: CARE criticality counter ≥ 72 (12 h)
- AE: d-256-64-16-64-256-d, ReLU, linear output, Adam 1e-3, batch 256, ≤40 epochs,
  early stop patience 5, seeds 0/1/2; sigmoid output run once as the ablation
