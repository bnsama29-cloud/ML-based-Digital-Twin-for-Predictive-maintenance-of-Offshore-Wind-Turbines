"""Benchmark on all 95 CARE-to-Compare datasets (farms A, B, C).

For every dataset it trains the detectors, turns each raw score into four variants
(raw, smoothed, condition-normalised, both) and stores the prediction-window scores row by
row, so every metric (CARE score, detection counts, ...) is computed afterwards by
merge_results.py without retraining.

Normally called by run_all.py. Resumable: datasets with an existing scores file are skipped.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import wtpm_lib as L

AE_SEEDS = (0, 1, 2)
KEEP_ON_DISK = {44, 61}   # analyze_44.py needs these afterwards


def run_dataset(eid, farm, ev, feats, a):
    df = L.load_dataset(eid, a.data_dir, farm)
    prep = L.prepare(df, feats, farm=farm, train_filter=True)       # main pipeline
    prep_nf = L.prepare(df, feats, farm=farm, train_filter=False)   # ablation: no filter
    X = prep.X
    col_sub = L.column_subsystems(prep.columns, feats)
    bases, logs, chans = {}, [], []

    for seed in AE_SEEDS:
        errfn, log = L.train_autoencoder(X[prep.fit_idx], X[prep.val_idx], seed=seed,
                                         epochs=a.epochs)
        ch = errfn(X)
        bases[f"ae_s{seed}"] = (ch.mean(axis=1), prep)
        logs += [dict(event_id=eid, base=f"ae_s{seed}", epoch=e, train=t, val=v) for e, t, v in log]
        if ev["event_label"] == "anomaly":
            r = L.channel_ratios(ch, prep, ev)
            if r is not None:
                chans.append(pd.DataFrame(dict(event_id=eid, farm=farm, seed=seed,
                                               column=prep.columns, ratio=r,
                                               subsystem=[col_sub[c] for c in prep.columns])))
    errfn, log = L.train_autoencoder(prep_nf.X[prep_nf.fit_idx], prep_nf.X[prep_nf.val_idx],
                                     seed=0, epochs=a.epochs)
    bases["ae_nofilter_s0"] = (errfn(prep_nf.X).mean(axis=1), prep_nf)
    logs += [dict(event_id=eid, base="ae_nofilter_s0", epoch=e, train=t, val=v) for e, t, v in log]
    bases["iforest"] = (L.score_isolation_forest(X[prep.fit_idx], X), prep)
    bases["pca"] = (L.score_pca(X[prep.fit_idx], X), prep)
    bases["three_sigma"] = (L.score_three_sigma(X), prep)

    pm = prep.pred_mask
    ids = df["id"].to_numpy()
    store = dict(ids=ids[pm], eval_normal=prep.eval_normal[pm],
                 in_window=(ids[pm] >= ev["event_start_id"]) & (ids[pm] <= ev["event_end_id"]))
    thresholds = {}
    for base, (score, p) in bases.items():
        for var, s in L.variants(score, p).items():
            key = f"{base}|{var}"
            thr = float(np.quantile(s[p.cal_b], L.THRESHOLD_Q))
            store[key] = s[pm].astype(np.float32)
            thresholds[key] = thr
    meta = dict(event_id=int(eid), farm=farm, label=ev["event_label"],
                n_features=len(prep.columns), n_fit=int(len(prep.fit_idx)),
                n_filtered=int(prep.n_filtered), thresholds=thresholds)
    return store, meta, logs, chans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", nargs="+", default=list(L.FARMS))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--events", type=int, nargs="*", help="run only these event ids")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--keep-files", action="store_true", help="do not delete downloaded csv")
    a = ap.parse_args()
    sdir = os.path.join(a.out, "scores")
    os.makedirs(sdir, exist_ok=True)

    for farm in a.farms:
        events = L.load_event_info(a.data_dir, farm).set_index("event_id")
        feats = L.load_features(a.data_dir, farm)
        ids = [e for e in events.index if not a.events or e in a.events]
        for eid in ids:
            npz = os.path.join(sdir, f"{farm}_{eid}.npz")
            if os.path.exists(npz):
                continue
            t0 = time.time()
            ev = events.loc[eid].copy()
            ev["event_id"] = eid
            store, meta, logs, chans = run_dataset(eid, farm, ev, feats, a)
            for path, rows in ((os.path.join(a.out, "trainlog.csv"), pd.DataFrame(logs)),
                               (os.path.join(a.out, "channels.csv"),
                                pd.concat(chans) if chans else pd.DataFrame())):
                if len(rows):
                    rows.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            with open(os.path.join(sdir, f"{farm}_{eid}.json"), "w") as f:
                json.dump(meta, f)
            np.savez_compressed(npz, **store)        # written last: marks the dataset done
            crit = L.criticality(store["ae_s0|cn_smooth"] > meta["thresholds"]["ae_s0|cn_smooth"],
                                 store["eval_normal"]).max()
            print(f"farm {farm} event {eid} ({meta['label']}): {meta['n_features']} features, "
                  f"{meta['n_filtered']} idle rows filtered, max criticality {crit}, "
                  f"{time.time() - t0:.0f}s", flush=True)
            if not a.keep_files and eid not in KEEP_ON_DISK:
                f = L.dataset_path(eid, a.data_dir, farm)
                if os.path.exists(f):
                    os.remove(f)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
