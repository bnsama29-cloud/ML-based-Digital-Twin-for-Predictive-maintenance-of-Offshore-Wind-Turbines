"""Benchmark on all 95 CARE-to-Compare datasets (farms A, B, C).

For every dataset it trains the detectors, turns each raw score into four variants
(raw, smoothed, condition-normalised, both) and stores, row by row, the prediction-window
scores and the calibration-slice scores, so every metric (CARE score, detection counts,
threshold tuning, ...) is computed afterwards by merge_results.py without retraining.

v3 detectors: AE (3 seeds), AE without the training filter, conditional AE (3 seeds),
AE with a sigmoid output head (seed 0), Isolation Forest (3 seeds), PCA, 3-sigma fraction.
Robustness: the prediction window is corrupted (noise / missing values) and rescored by
AE seed 0 and PCA without retraining.
Run B (--detectors lstm): LSTM AE (3 seeds) and compute times.
Run C (--detectors polish): AE, condition model (NBM), IF (3 seeds each), PCA, 3-sigma, every
score also post-processed with EWMA and CUSUM, plus sensor-fault tests (drift, stuck, dropout).
Run D (--detectors arch): the AE in 6 configurations (3 sizes x 2 learning rates) for the
architecture search, analysed by arch_select.py.
Run E (--detectors thermal): physics-structured thermal model of component temperatures and
the AE (seed 0), with attribution over the same temperature channels.

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
KEEP_ON_DISK = {44, 61, 40}   # case studies need these afterwards (ids unique across farms)
ROBUST_BASES = ("ae_s0", "pca")
ROBUST_VARIANTS = ("raw", "smooth", "cn_smooth")


def run_dataset_lstm(eid, farm, ev, feats, a):
    """Run B: LSTM autoencoder only (3 seeds), same preprocessing, splits and storage."""
    df = L.load_dataset(eid, a.data_dir, farm)
    prep = L.prepare(df, feats, farm=farm, train_filter=True)
    bases, logs = {}, []
    X = prep.X
    for seed in AE_SEEDS:
        score, log = L.train_lstm_autoencoder(X, prep.fit_idx, prep.val_idx, seed=seed,
                                              epochs=a.epochs)
        bases[f"lstm_s{seed}"] = (score, prep)
        logs += [dict(event_id=eid, base=f"lstm_s{seed}", epoch=e, train=t, val=v)
                 for e, t, v in log]
    timing = time_detectors(X, prep, a.epochs)
    timing = [dict(event_id=eid, farm=farm, n_features=X.shape[1], **t) for t in timing]
    return store_scores(df, prep, ev, eid, farm, bases), logs, timing


def time_detectors(X, prep, epochs):
    """Wall-clock fit time and scoring time per 1000 rows of every detector (seed 0)."""
    def clock(fn):
        t0 = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - t0

    rows = []
    Xf, Xv = X[prep.fit_idx], X[prep.val_idx]
    (fn, _), t_fit = clock(lambda: L.train_autoencoder(Xf, Xv, seed=0, epochs=epochs))
    _, t_inf = clock(lambda: fn(X))
    rows.append(dict(detector="ae", fit_s=t_fit, score_s_per_1k=1000 * t_inf / len(X)))
    _, t_fit = clock(lambda: L.train_lstm_autoencoder(X, prep.fit_idx, prep.val_idx, seed=0,
                                                      epochs=epochs))
    rows.append(dict(detector="lstm", fit_s=t_fit, score_s_per_1k=np.nan))
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    for name, model in (("pca", PCA(n_components=0.95, svd_solver="full")),
                        ("iforest", IsolationForest(n_estimators=200, random_state=0, n_jobs=-1))):
        m, t_fit = clock(lambda: model.fit(Xf))
        score = (lambda: m.inverse_transform(m.transform(X))) if name == "pca" \
            else (lambda: m.score_samples(X))
        _, t_inf = clock(score)
        rows.append(dict(detector=name, fit_s=t_fit, score_s_per_1k=1000 * t_inf / len(X)))
    _, t_inf = clock(lambda: L.score_three_sigma(X))
    rows.append(dict(detector="three_sigma", fit_s=0.0, score_s_per_1k=1000 * t_inf / len(X)))
    return rows


def store_scores(df, prep, ev, eid, farm, bases):
    pm = prep.pred_mask
    ids = df["id"].to_numpy()
    store = dict(ids=ids[pm], eval_normal=prep.eval_normal[pm],
                 in_window=(ids[pm] >= ev["event_start_id"]) & (ids[pm] <= ev["event_end_id"]))
    thresholds = {}
    for base, (score, p) in bases.items():
        for var, s in L.variants(score, p).items():
            key = f"{base}|{var}"
            store[key] = s[pm].astype(np.float32)
            store[key + "@cal"] = s[p.cal_b].astype(np.float32)
            thresholds[key] = float(np.quantile(s[p.cal_b], L.THRESHOLD_Q))
    meta = dict(event_id=int(eid), farm=farm, label=ev["event_label"],
                n_features=len(prep.columns), n_fit=int(len(prep.fit_idx)),
                n_filtered=int(prep.n_filtered), thresholds=thresholds)
    return store, meta


def run_dataset_thermal(eid, farm, ev, feats, a):
    """Run E: physics-structured thermal model of every component temperature next to the AE
    (seed 0); both attribute faults over the same temperature channels. Also the AE trained on
    only the most recent 25 % / 50 % of the fit rows (how much healthy history is needed), and an
    immune-inspired negative-selection detector (3 seeds)."""
    df = L.load_dataset(eid, a.data_dir, farm)
    prep = L.prepare(df, feats, farm=farm, train_filter=True)
    X = prep.X
    col_sub = L.column_subsystems(prep.columns, feats)
    op_sensors = {L.base_sensor(c) for c in L.OPERATING[farm].values()}
    tc = L.temperature_channels(prep.columns, feats, col_sub, exclude=op_sensors)
    bases, logs, ch_th, ch_ae = {}, [], [], []
    errfn, log = L.train_autoencoder(X[prep.fit_idx], X[prep.val_idx], seed=0, epochs=a.epochs)
    ch = errfn(X)
    bases["ae_s0"] = (ch.mean(axis=1), prep)
    logs += [dict(event_id=eid, base="ae_s0", epoch=e, train=t, val=v) for e, t, v in log]
    # how much healthy history is needed: AE on the most recent part of the fit rows only
    for frac in L.HISTORY_FRACTIONS:
        sub = L.recent_rows(prep.fit_idx, frac)
        hfn, log = L.train_autoencoder(X[sub], X[prep.val_idx], seed=0, epochs=a.epochs)
        name = f"ae_hist{int(frac * 100)}_s0"
        bases[name] = (hfn(X).mean(axis=1), prep)
        logs += [dict(event_id=eid, base=name, epoch=e, train=t, val=v) for e, t, v in log]
    for seed in (0, 1, 2):     # immune-inspired negative selection (bio-inspired baseline)
        bases[f"nsa_s{seed}"] = (L.score_negative_selection(X[prep.fit_idx], X, seed=seed), prep)
    if tc:
        pred = L.fit_thermal(L.thermal_inputs(prep), X[:, tc], prep.fit_idx)
        th = (X[:, tc] - pred) ** 2
        bases["thermal"] = (th.mean(axis=1), prep)
        bases["ae_temp_s0"] = (ch[:, tc].mean(axis=1), prep)
        if ev["event_label"] == "anomaly":
            cols = [prep.columns[i] for i in tc]
            subs = [col_sub[c] for c in cols]
            for target, err in ((ch_th, th), (ch_ae, ch[:, tc])):
                r = L.channel_ratios(err, prep, ev)
                if r is not None:
                    target.append(pd.DataFrame(dict(event_id=eid, farm=farm, seed=0, column=cols,
                                                    ratio=r, subsystem=subs)))
    store, meta = store_scores(df, prep, ev, eid, farm, bases)
    meta["n_temperature_channels"] = len(tc)
    return store, meta, logs, (ch_th, ch_ae)


def run_dataset_arch(eid, farm, ev, feats, a):
    """Run D (architecture search): the AE in six fixed configurations (3 sizes x 2 learning
    rates, seed 0). The training log keeps every epoch's validation loss for label-free
    selection; scores are stored like every other detector."""
    df = L.load_dataset(eid, a.data_dir, farm)
    prep = L.prepare(df, feats, farm=farm, train_filter=True)
    X = prep.X
    bases, logs = {}, []
    for name, (widths, lr) in L.ARCH_CONFIGS.items():
        errfn, log = L.train_autoencoder(X[prep.fit_idx], X[prep.val_idx], seed=0, lr=lr,
                                         epochs=a.epochs, widths=widths)
        bases[f"{name}_s0"] = (errfn(X).mean(axis=1), prep)
        logs += [dict(event_id=eid, farm=farm, base=f"{name}_s0", epoch=e, train=t, val=v)
                 for e, t, v in log]
    store, meta = store_scores(df, prep, ev, eid, farm, bases)
    return store, meta, logs


def run_dataset_polish(eid, farm, ev, feats, a):
    """Run C (polish): AE (3 seeds), multi-target condition model NBM (3 seeds), Isolation
    Forest (3 seeds), PCA and 3-sigma, every score also post-processed with EWMA and CUSUM;
    sensor-fault tests (drift, stuck, dropout) on AE seed 0 and NBM seed 0."""
    df = L.load_dataset(eid, a.data_dir, farm)
    prep = L.prepare(df, feats, farm=farm, train_filter=True)
    X = prep.X
    U = L.scaled_conditions(prep)
    col_sub = L.column_subsystems(prep.columns, feats)
    subs = np.array([col_sub[c] for c in prep.columns])
    bases, logs, chans, nbm_chans, faults = {}, [], [], [], []
    fns = {}

    def keep_ratios(target, ch, name, seed):
        if ev["event_label"] != "anomaly":
            return
        r = L.channel_ratios(ch, prep, ev)
        if r is not None:
            target.append(pd.DataFrame(dict(event_id=eid, farm=farm, seed=seed,
                                            column=prep.columns, ratio=r, subsystem=subs,
                                            detector=name)))

    for seed in AE_SEEDS:
        errfn, log = L.train_autoencoder(X[prep.fit_idx], X[prep.val_idx], seed=seed,
                                         epochs=a.epochs)
        ch = errfn(X)
        bases[f"ae_s{seed}"] = (ch.mean(axis=1), prep)
        keep_ratios(chans, ch, "ae", seed)
        logs += [dict(event_id=eid, base=f"ae_s{seed}", epoch=e, train=t, val=v) for e, t, v in log]
        if seed == 0:
            fns["ae_s0"] = lambda Z, f=errfn: f(Z)
        nfn, log = L.train_condition_model(U[prep.fit_idx], X[prep.fit_idx], U[prep.val_idx],
                                           X[prep.val_idx], seed=seed, epochs=a.epochs)
        ch = nfn(U, X)
        bases[f"nbm_s{seed}"] = (ch.mean(axis=1), prep)
        keep_ratios(nbm_chans, ch, "nbm", seed)
        logs += [dict(event_id=eid, base=f"nbm_s{seed}", epoch=e, train=t, val=v) for e, t, v in log]
        if seed == 0:
            fns["nbm_s0"] = lambda Z, f=nfn: f(U, Z)
        bases[f"iforest_s{seed}"] = (L.score_isolation_forest(X[prep.fit_idx], X, seed=seed), prep)
    bases["pca"] = (L.score_pca(X[prep.fit_idx], X), prep)
    bases["three_sigma"] = (L.score_three_sigma(X), prep)

    pm = prep.pred_mask
    ids = df["id"].to_numpy()
    store = dict(ids=ids[pm], eval_normal=prep.eval_normal[pm],
                 in_window=(ids[pm] >= ev["event_start_id"]) & (ids[pm] <= ev["event_end_id"]))
    thresholds = {}
    for base, (score, p) in bases.items():
        for var, s in L.variants(score, p, rules=True).items():
            key = f"{base}|{var}"
            store[key] = s[pm].astype(np.float32)
            store[key + "@cal"] = s[p.cal_b].astype(np.float32)
            thresholds[key] = float(np.quantile(s[p.cal_b], L.THRESHOLD_Q))

    # sensor-fault tests: corrupt the prediction window, rescore without retraining
    op_sensors = {L.base_sensor(c) for c in L.OPERATING[farm].values()}
    ch_i = L.pick_fault_channel(prep.columns, col_sub, eid, exclude=op_sensors)
    pm_rows = np.flatnonzero(pm)
    eval_rows = pm_rows[prep.eval_normal[pm_rows]]
    if ch_i is not None and len(eval_rows):
        area = subs[ch_i]
        area_cols = np.flatnonzero(subs == area)
        fill = np.median(X[prep.fit_idx], axis=0)
        for kind in ("none", *L.SENSOR_FAULTS):
            Xc = X if kind == "none" else L.sensor_fault(X, pm_rows, kind, ch_i, area_cols, fill)
            for base, fn in fns.items():
                ch = fn(Xc)
                if kind != "none":
                    for var, s in L.variants(ch.mean(axis=1), prep).items():
                        if var == "cn_smooth":
                            store[f"rob:{kind}:{base}|{var}"] = s[pm].astype(np.float32)
                r = L.window_ratios(ch, prep, eval_rows)
                order = np.argsort(r)[::-1]
                ranked = L.rank_subsystems(r, subs, "vote")
                faults.append(dict(event_id=eid, farm=farm, label=ev["event_label"], kind=kind,
                                   base=base, column=prep.columns[ch_i], area=area,
                                   n_area_channels=len(area_cols),
                                   channel_rank=int(np.flatnonzero(order == ch_i)[0]) + 1,
                                   area_rank=(ranked.index(area) + 1) if area in ranked else np.nan,
                                   n_channels=len(prep.columns)))
    meta = dict(event_id=int(eid), farm=farm, label=ev["event_label"],
                n_features=len(prep.columns), n_fit=int(len(prep.fit_idx)),
                n_filtered=int(prep.n_filtered), thresholds=thresholds)
    return store, meta, logs, (chans, nbm_chans, faults)


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
        if seed == 0:
            ae0_fn = errfn
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

    # conditional AE: operating conditions go into encoder and decoder
    U = L.scaled_conditions(prep)
    for seed in AE_SEEDS:
        cfn, log = L.train_autoencoder(X[prep.fit_idx], X[prep.val_idx], seed=seed,
                                       epochs=a.epochs, U_fit=U[prep.fit_idx],
                                       U_val=U[prep.val_idx])
        bases[f"cae_s{seed}"] = (cfn(X, U).mean(axis=1), prep)
        logs += [dict(event_id=eid, base=f"cae_s{seed}", epoch=e, train=t, val=v)
                 for e, t, v in log]

    # CSITSS reviewer 5: sigmoid head on standardised inputs
    sfn, log = L.train_autoencoder(X[prep.fit_idx], X[prep.val_idx], seed=0, epochs=a.epochs,
                                   output="sigmoid")
    bases["ae_sigmoid_s0"] = (sfn(X).mean(axis=1), prep)
    logs += [dict(event_id=eid, base="ae_sigmoid_s0", epoch=e, train=t, val=v) for e, t, v in log]

    for seed in AE_SEEDS:
        bases[f"iforest_s{seed}"] = (L.score_isolation_forest(X[prep.fit_idx], X, seed=seed), prep)
    pca_fn = lambda Z: L.score_pca(X[prep.fit_idx], Z)  # noqa: E731
    bases["pca"] = (pca_fn(X), prep)
    bases["three_sigma"] = (L.score_three_sigma(X), prep)

    if ev["event_label"] == "anomaly":
        rz = L.zscore_ratios(X, prep, ev)
        if rz is not None:
            chans.append(pd.DataFrame(dict(event_id=eid, farm=farm, seed=-1, column=prep.columns,
                                           ratio=rz, subsystem=[col_sub[c] for c in prep.columns])))

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
            store[key + "@cal"] = s[p.cal_b].astype(np.float32)
            thresholds[key] = thr

    # robustness: corrupt only the prediction window, rescore, keep the clean threshold
    fill = np.median(X[prep.fit_idx], axis=0)
    pm_rows = np.flatnonzero(pm)
    for kind, level in L.CORRUPTIONS:
        Xc = L.corrupt(X, pm_rows, kind, level, fill, seed=0)
        rob = {"ae_s0": ae0_fn(Xc).mean(axis=1), "pca": pca_fn(Xc)}
        for base in ROBUST_BASES:
            vs = L.variants(rob[base], prep)
            for var in ROBUST_VARIANTS:
                store[f"rob:{kind}{level}:{base}|{var}"] = vs[var][pm].astype(np.float32)
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
    ap.add_argument("--detectors", choices=["main", "lstm", "polish", "arch", "thermal"],
                    default="main",
                    help="main = run A (all v3 detectors); lstm = run B (LSTM autoencoder only); "
                         "polish = run C (AE, NBM, IF, PCA, 3-sigma with EWMA/CUSUM rules and "
                         "sensor-fault tests)")
    ap.add_argument("--budget-h", type=float, default=0,
                    help="stop starting new datasets after this many hours (0 = no limit)")
    a = ap.parse_args()
    t_start = time.time()
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
            if a.budget_h and (time.time() - t_start) / 3600 > a.budget_h:
                print(f"time budget of {a.budget_h} h reached: skipping farm {farm} "
                      f"event {eid} and the rest", flush=True)
                break
            t0 = time.time()
            ev = events.loc[eid].copy()
            ev["event_id"] = eid
            if a.detectors == "lstm":
                (store, meta), logs, chans = run_dataset_lstm(eid, farm, ev, feats, a)
                sides = [(os.path.join(a.out, "timing.csv"), pd.DataFrame(chans))]
            elif a.detectors == "thermal":
                store, meta, logs, (ch_th, ch_ae) = run_dataset_thermal(eid, farm, ev, feats, a)
                sides = [(os.path.join(a.out, "channels_thermal.csv"),
                          pd.concat(ch_th) if ch_th else pd.DataFrame()),
                         (os.path.join(a.out, "channels_ae_temp.csv"),
                          pd.concat(ch_ae) if ch_ae else pd.DataFrame())]
            elif a.detectors == "arch":
                store, meta, logs = run_dataset_arch(eid, farm, ev, feats, a)
                sides = []
            elif a.detectors == "polish":
                store, meta, logs, (chans, nbm_chans, faults) = run_dataset_polish(
                    eid, farm, ev, feats, a)
                sides = [(os.path.join(a.out, "channels.csv"),
                          pd.concat(chans) if chans else pd.DataFrame()),
                         (os.path.join(a.out, "channels_nbm.csv"),
                          pd.concat(nbm_chans) if nbm_chans else pd.DataFrame()),
                         (os.path.join(a.out, "sensorfault.csv"), pd.DataFrame(faults))]
            else:
                store, meta, logs, chans = run_dataset(eid, farm, ev, feats, a)
                sides = [(os.path.join(a.out, "channels.csv"),
                          pd.concat(chans) if chans else pd.DataFrame())]
            for path, rows in ((os.path.join(a.out, "trainlog.csv"), pd.DataFrame(logs)), *sides):
                if len(rows):
                    rows.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            with open(os.path.join(sdir, f"{farm}_{eid}.json"), "w") as f:
                json.dump(meta, f)
            np.savez_compressed(npz, **store)        # written last: marks the dataset done
            k0 = {"lstm": "lstm_s0|cn_smooth", "arch": f"{L.ARCH_REFERENCE}_s0|cn_smooth"}.get(
                a.detectors, "ae_s0|cn_smooth")
            crit = L.criticality(store[k0] > meta["thresholds"][k0], store["eval_normal"]).max()
            print(f"farm {farm} event {eid} ({meta['label']}): {meta['n_features']} features, "
                  f"{meta['n_filtered']} idle rows filtered, max criticality {crit}, "
                  f"{time.time() - t0:.0f}s", flush=True)
            if not a.keep_files and (a.detectors != "main" or eid not in KEEP_ON_DISK):
                f = L.dataset_path(eid, a.data_dir, farm)
                if os.path.exists(f):
                    os.remove(f)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
