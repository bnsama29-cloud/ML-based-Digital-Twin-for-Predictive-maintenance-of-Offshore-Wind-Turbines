# Running the rework on Kaggle (3 accounts, one per teammate)

Each teammate runs **one shard** on their own Kaggle account. Shards split the 58 farm C
datasets (26 anomaly, 32 normal) about 20 each. Measured locally: one dataset takes
about 2.5 min on a 2-core CPU (4 autoencoders + Isolation Forest + PCA), plus the
~300 MB download from Zenodo. Expect roughly 1–2 h per shard. A GPU helps the
autoencoders but is not required.

## Notebook settings
- Settings → Internet **on** (needed to stream files from Zenodo; needs a phone-verified account)
- Accelerator: GPU T4 if available, otherwise CPU is fine
- Use **Save Version → Save & Run All (Commit)** so it keeps running after you close the tab

## Cell 1 — setup (same for everyone)
```python
!pip -q install remotezip
!git clone -q --depth 1 -b rework-2026 https://github.com/bnsama29-cloud/ml-based-digital-twin-for-predictive-maintenance-of-offshore-wind-turbines.git repo
%cd repo/rework
```

## Cell 2 — the shard (change only SHARD)
```python
SHARD = 0   # Samarth = 0, Ananth = 1, Roshan = 2
!python run_benchmark.py --shard {SHARD} --n-shards 3 --data-dir /tmp/care --out /kaggle/working/shard{SHARD}
```
If the run stops halfway, run it again: finished datasets are skipped.

## Cell 3 — only on shard 0: the dataset-44 deep dive (CPU, ~10 min)
```python
!python analyze_44.py --data-dir /tmp/care --out /kaggle/working/deep44
```

## After all three finish
Download `/kaggle/working/shard*/` from each account (Output tab), put the three folders
side by side, then:
```bash
python merge_results.py --inputs shard0 shard1 shard2 --out paper_tables
```
`paper_tables/detection.tex` and `attribution.tex` go into the paper; `deep44/` holds the
figures and CSVs for Section V-B to V-E.

## Rules fixed before looking at results (do not change after seeing them)
- training rows: `train_test == train` and status 0/2; last 15 % = validation
- point threshold: 99th percentile of validation score
- alarm: CARE criticality counter ≥ 72 (12 h)
- AE: d-256-64-16-64-256-d, ReLU, linear output, Adam 1e-3, batch 256, ≤40 epochs,
  early stop patience 5, seeds 0/1/2; sigmoid output run once as the ablation
