"""Detection + attribution benchmark on all 58 CARE farm C datasets, split into shards.

Normally called by run_all.py (one Kaggle account does everything).
Sharding is optional: --n-shards N --shard k splits the 58 datasets if you ever want to.
Resumable: datasets already in results_shard<k>.csv are skipped.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import wtpm_lib as L

AE_SEEDS = (0, 1, 2)
KEEP_ON_DISK = {44, 61}   # analyze_44.py needs these afterwards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--events", type=int, nargs="*", help="override: run only these event ids")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--keep-files", action="store_true", help="do not delete downloaded csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.data_dir, exist_ok=True)

    events = L.load_event_info(a.data_dir)
    feats = L.load_features(a.data_dir)
    ids = a.events or [e for i, e in enumerate(events["event_id"]) if i % a.n_shards == a.shard]
    res_path = os.path.join(a.out, f"results_shard{a.shard}.csv")
    rca_path = os.path.join(a.out, f"rca_shard{a.shard}.csv")
    log_path = os.path.join(a.out, f"trainlog_shard{a.shard}.csv")
    done = set(pd.read_csv(res_path)["event_id"]) if os.path.exists(res_path) else set()
    print(f"shard {a.shard}: {len(ids)} datasets, {len(done)} already done")

    for eid in ids:
        if eid in done:
            continue
        t0 = time.time()
        ev = events.set_index("event_id").loc[eid].copy()
        ev["event_id"] = eid
        df = L.load_dataset(eid, a.data_dir)
        prep = L.prepare(df, feats)
        Xf, Xv, X = prep.X[prep.fit_idx], prep.X[prep.val_idx], prep.X
        col_sub = L.column_subsystems(prep.columns, feats)
        recs, rcas, logs = [], [], []

        def add(method, seed, score):
            thr = float(np.quantile(score[prep.val_idx], L.THRESHOLD_Q))
            r = L.evaluate(score, prep, ev, thr)
            r.update(method=method, seed=seed, threshold=thr, n_features=len(prep.columns),
                     n_fit=len(prep.fit_idx))
            recs.append(r)

        for seed in AE_SEEDS:
            errfn, log = L.train_autoencoder(Xf, Xv, seed=seed, epochs=a.epochs)
            ch = errfn(X)
            add("ae_linear", seed, ch.mean(axis=1))
            logs += [dict(event_id=eid, method="ae_linear", seed=seed, epoch=e, train=t, val=v)
                     for e, t, v in log]
            if ev["event_label"] == "anomaly":
                s = L.subsystem_scores(ch, prep, ev, col_sub)
                if len(s):
                    rcas.append(s.assign(event_id=eid, seed=seed))
        errfn, log = L.train_autoencoder(Xf, Xv, seed=0, output="sigmoid", epochs=a.epochs)
        ch = errfn(X)
        add("ae_sigmoid", 0, ch.mean(axis=1))
        logs += [dict(event_id=eid, method="ae_sigmoid", seed=0, epoch=e, train=t, val=v)
                 for e, t, v in log]
        add("isolation_forest", 0, L.score_isolation_forest(Xf, X))
        add("pca", 0, L.score_pca(Xf, X))
        add("three_sigma", 0, L.score_three_sigma(X))

        for path, rows in ((res_path, pd.DataFrame(recs)), (log_path, pd.DataFrame(logs)),
                           (rca_path, pd.concat(rcas) if rcas else pd.DataFrame())):
            if len(rows):
                rows.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"event {eid} ({ev['event_label']}): {len(prep.columns)} features, "
              f"{time.time() - t0:.0f}s")
        if not a.keep_files and eid not in KEEP_ON_DISK:
            f = os.path.join(a.data_dir, L.FARM_PREFIX, "datasets", f"{eid}.csv")
            if os.path.exists(f):
                os.remove(f)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
