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

## Follow-up run C (`followup_run_c/`)
`run_all.py --detectors polish` on Kaggle (7.47 h, no failed steps, all 95 datasets): AE and the
multi-target normal behaviour model (NBM) with 3 seeds each, Isolation Forest (3 seeds), PCA and
3-sigma; every score also with EWMA and CUSUM alarm rules (variants `raw_ewma`, `raw_cusum`,
`cn_ewma`, `cn_cusum`); sensor-fault tests on AE and NBM seed 0.

| File | Content |
|---|---|
| `care.csv`, `bootstrap.csv`, `lofo.csv`, `counter_sensitivity.csv` | as above, including the new alarm rules and the NBM |
| `attribution_nbm.csv` | plant-area attribution from NBM residual ratios (40 labelled faults) |
| `sensorfault.csv` | normal datasets only: share of cases where the corrupted channel is ranked first / in the top 10, and its plant area first / in the top 3 |
| `sensorfault_rows.csv` | one row per dataset × fault kind × model |
| `robustness.csv` | CARE, detections and false alarms under drift, stuck and dropout faults |

The AE was retrained in this run and moves by at most 0.009 CARE against run A (GPU
non-determinism); IF, PCA and 3-sigma reproduce run A to four decimals.

## Follow-up run E (`followup_run_e/`)
`--detectors thermal` (3.64 h, no failed steps): physics-structured thermal model (`thermal`), AE
seed 0 on all channels (`ae`) and on temperature channels only (`ae_temp`), AE trained on the
latest 25 % / 50 % of fit rows (`ae_hist25`, `ae_hist50`), immune-inspired negative selection
(`nsa`, 3 seeds). `complement.csv`: AE, thermal model, "either" rule and faults found by only one.

## Follow-up run G (`followup_run_g/`)
`--detectors final` (4.03 h, no failed steps): architecture search (6 AE configurations, seed 0)
and AE / IF / PCA / 3-sigma on average values only (`*_avg`). The rows "val per farm" and
"val per dataset" in `arch_search.csv` / `arch_bootstrap.csv` were produced before a fix: the
label-free choice could pick the average-only AE, which is not part of the search. The corrected
rows, recomputed from `per_event.csv` and the training log with the fixed `arch_select.py`, are in
`arch_val_selection_corrected.csv`; all other rows are unaffected.
