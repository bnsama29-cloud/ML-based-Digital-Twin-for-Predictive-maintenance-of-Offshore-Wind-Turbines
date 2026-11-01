# Results of the final run

Produced by `pipeline/run_all.py` on Kaggle (run A: 5.26 h, run B: 4.84 h, no failed steps;
see `run_info_A.txt`, `run_info_B.txt`) and merged with `pipeline/merge_results.py`.
All 95 CARE-to-Compare datasets finished in both runs.

| Path | Content |
|---|---|
| `tables/care.csv`, `care.tex` | CARE score, coverage, accuracy, reliability, earliness, detections and false alarms for every detector × post-processing; `farm` = all / A / B / C; mean and s.d. over seeds |
| `tables/bootstrap.csv` | paired bootstrap (1000 draws, fault and normal datasets resampled separately) of CARE differences |
| `tables/lofo.csv` | leave-one-farm-out tuning of threshold quantile and alarm counter; `care_insample_best` is an optimistic ceiling tuned on the test data |
| `tables/counter_sensitivity.csv` | alarm counter of 36, 72 and 144 rows |
| `tables/robustness.csv` | AE (seed 0) and PCA under Gaussian noise and missing values, thresholds unchanged |
| `tables/val_loss_heads.csv` | best validation loss per dataset for linear vs sigmoid output, conditional AE and LSTM AE |
| `tables/attribution.csv`, `attribution.tex` | top-1 / top-3 plant-area hit rates on 40 labelled faults, with random, majority-class and z-score floors |
| `tables/timing.csv` | median fit time and scoring time per 1000 rows, per detector and farm (T4 GPU) |
| `tables/per_event.csv` | one row per dataset × detector × seed × post-processing |
| `case_40/` | generator-bearing fault (farm A): detection summary, gate sensitivity, attribution |
| `case_44/` | cooling-water valve fault (farm C): the same plus the temperature normal-behaviour model |

Figures drawn from these files are in [`../figures`](../figures). The case-40 figures there
were redrawn with a local CPU rerun of `analyze_case.py` (first alarm 46.0 h after the
window start, against 45.2 h in the Kaggle run whose numbers are in `case_40/`).

Note on counts: in Zenodo record 15846963, `event_info.csv` marks 45 datasets as anomaly and
50 as normal (the CARE article reports 44 / 51).
