"""Turn the stored scores into the paper tables: CARE score, detection, attribution,
leave-one-farm-out threshold tuning, paired bootstrap, robustness, sigmoid-head ablation.

    python merge_results.py --inputs results/benchmark --out results/paper_tables
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import wtpm_lib as L

BASE_LABEL = {"ae": "Autoencoder", "ae_nofilter": "AE, no training filter",
              "cae": "Conditional AE", "ae_sigmoid": "AE, sigmoid output", "lstm": "LSTM AE",
              "iforest": "Isolation Forest", "pca": "PCA", "three_sigma": "3-sigma fraction"}
VAR_LABEL = {"raw": "raw", "smooth": "+ smoothing", "cn": "+ condition norm.",
             "cn_smooth": "+ cond. norm. + smoothing"}


def load_raw(inputs):
    """[(meta, arrays)] for every finished dataset. Several input folders (run A + run B)
    are combined per dataset; the row arrays must agree, which a shared preprocessing
    guarantees, and this is checked."""
    by = {}
    for d in inputs:
        for js in sorted(glob.glob(os.path.join(d, "scores", "*.json"))):
            npz = js[:-5] + ".npz"
            if not os.path.exists(npz):
                continue
            meta, data = json.load(open(js)), dict(np.load(npz))
            key = (meta["farm"], meta["event_id"])
            if key not in by:
                by[key] = (meta, data)
                continue
            m0, d0 = by[key]
            for k in ("ids", "eval_normal", "in_window"):
                if not np.array_equal(d0[k], data[k]):
                    raise ValueError(f"rows of dataset {key} differ between input folders")
            m0["thresholds"].update(meta["thresholds"])
            d0.update({k: v for k, v in data.items() if k not in d0})
    return [by[k] for k in sorted(by)]


def load_events(raw):
    rows = []
    for meta, data in raw:
        for key, thr in meta["thresholds"].items():
            base, var = key.split("|")
            r = L.care_event(data[key] > thr, data["eval_normal"], data["in_window"],
                             meta["label"])
            group, seed = L.split_key(base)
            rows.append(dict(event_id=meta["event_id"], farm=meta["farm"], method=group,
                             seed=seed, variant=var, **r))
    return pd.DataFrame(rows)


def care_table(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, var), g in ev.groupby(["method", "variant"]):
        for farm in ["all", *sorted(g["farm"].unique())]:
            gf = g if farm == "all" else g[g["farm"] == farm]
            per_seed = pd.DataFrame([L.care_score(s) for _, s in gf.groupby("seed")])
            row = dict(method=method, variant=var, farm=farm, seeds=len(per_seed))
            for c in per_seed.columns:
                row[c] = per_seed[c].mean()
                row[c + "_sd"] = per_seed[c].std() if len(per_seed) > 1 else np.nan
            lo, hi = L.wilson(int(round(row["detected"])), int(row["n_anomaly"]))
            row["detected_ci"] = f"[{lo:.2f}, {hi:.2f}]"
            lo, hi = L.wilson(int(round(row["false_alarms"])), int(row["n_normal"]))
            row["false_alarm_ci"] = f"[{lo:.2f}, {hi:.2f}]"
            rows.append(row)
    return pd.DataFrame(rows)


def attribution_table(ch: pd.DataFrame) -> pd.DataFrame:
    """Top-1 / top-3 hit rate against the hand-labelled plant area of each anomaly event."""
    labelled = {e: s for e, s in L.EVENT_SUBSYSTEMS.items() if s}
    ch = ch[ch["event_id"].isin(labelled)].drop_duplicates(["event_id", "seed", "column"],
                                                           keep="last")
    zch = ch[ch["seed"] < 0]
    ch = ch[ch["seed"] >= 0]
    rows = []
    for farm in ["all", *sorted(ch["farm"].unique())]:
        cf = ch if farm == "all" else ch[ch["farm"] == farm]
        events = sorted(cf["event_id"].unique())
        if not events:
            continue
        for rule in ("mean", "sum", "vote"):
            per_seed = []
            for seed, g in cf.groupby("seed"):
                h1 = h3 = 0
                for eid, e in g.groupby("event_id"):
                    ranked = L.rank_subsystems(e["ratio"].to_numpy(), e["subsystem"].to_numpy(),
                                               rule)
                    truth = set(labelled[eid])
                    h1 += bool(truth & set(ranked[:1]))
                    h3 += bool(truth & set(ranked[:3]))
                per_seed.append((h1 / len(events), h3 / len(events)))
            a = np.array(per_seed)
            rows.append(dict(farm=farm, rule=f"AE ratio, {rule}", top1=a[:, 0].mean(),
                             top1_sd=a[:, 0].std(), top3=a[:, 1].mean(), top3_sd=a[:, 1].std(),
                             n_events=len(events)))
        zf = zch if farm == "all" else zch[zch["farm"] == farm]
        if len(zf):
            h1 = h3 = 0
            for eid, e in zf.groupby("event_id"):
                ranked = L.rank_subsystems(e["ratio"].to_numpy(), e["subsystem"].to_numpy(),
                                           "mean")
                h1 += bool(set(labelled[eid]) & set(ranked[:1]))
                h3 += bool(set(labelled[eid]) & set(ranked[:3]))
            n = zf["event_id"].nunique()
            rows.append(dict(farm=farm, rule="|z| ratio, mean (no model)", top1=h1 / n,
                             top3=h3 / n, n_events=n))
        # floors: the commonest true area (an oracle choice, it peeks at the labels) and random
        counts = pd.Series([s for e in events for s in labelled[e]]).value_counts()
        top = counts.index[:3].tolist()
        rows.append(dict(farm=farm, rule=f"always '{top[0]}' (oracle floor)",
                         top1=np.mean([top[0] in labelled[e] for e in events]),
                         top3=np.mean([bool(set(top) & set(labelled[e])) for e in events]),
                         n_events=len(events)))
        n_sub = {e: cf.loc[cf["event_id"] == e, "subsystem"].nunique() for e in events}
        rows.append(dict(farm=farm, rule="random ranking (expected)",
                         top1=np.mean([len(labelled[e]) / n_sub[e] for e in events]),
                         top3=np.mean([1 - np.prod([max(n_sub[e] - len(labelled[e]) - i, 0)
                                                    / (n_sub[e] - i) for i in range(3)])
                                       for e in events]),
                         n_events=len(events)))
    return pd.DataFrame(rows)


def per_event_setting(raw, key, q=None, steps=None):
    """Per-event CARE rows of one score key, threshold = quantile q of its calibration scores
    (default: the stored 99th-percentile threshold) and alarm counter `steps`."""
    rows = []
    for meta, data in raw:
        if key not in data:
            continue
        thr = meta["thresholds"][key] if q is None else float(np.quantile(data[key + "@cal"], q))
        r = L.care_event(data[key] > thr, data["eval_normal"], data["in_window"], meta["label"],
                         steps=steps)
        rows.append(dict(event_id=meta["event_id"], farm=meta["farm"], **r))
    return pd.DataFrame(rows)


def keys_of(raw, method, var):
    """All seed keys of a method, e.g. ('ae', 'smooth') -> ['ae_s0|smooth', ...]."""
    keys = {k for _, d in raw for k in d if "|" in k and "@" not in k and ":" not in k}
    return sorted(k for k in keys if k.endswith("|" + var) and L.split_key(k.split("|")[0])[0] == method)


MAIN_CONFIGS = [(m, v) for m in ("ae", "cae", "lstm", "pca", "iforest") for v in L.VARIANTS]


def lofo_table(raw):
    """Leave-one-farm-out tuning of (threshold quantile, alarm counter)."""
    if not raw or not any(k.endswith("@cal") for k in raw[0][1]):
        return pd.DataFrame()
    out = []
    for method, var in MAIN_CONFIGS:
        keys = keys_of(raw, method, var)
        if not keys:
            continue
        grid = []
        for q in L.TUNE_Q:
            for st in L.TUNE_STEPS:
                for k in keys:
                    ev = per_event_setting(raw, k, q, st)
                    for f, g in ev.groupby("farm"):
                        grid.append(dict(farm=f, q=q, steps=st, seed=L.split_key(k.split("|")[0])[1],
                                         events=g))
        grid = pd.DataFrame(grid)
        sel = L.lofo_select(grid)
        if sel.empty:                  # fewer than two farms finished for this detector
            continue
        cares = [L.care_score(g) for _, g in sel.groupby("seed")]
        fixed = [L.care_score(per_event_setting(raw, k)) for k in keys]
        # in-sample best (tuned on the test data; an optimistic ceiling, not a result)
        best = max(np.mean([L.care_score(pd.concat(g[g.seed == s]["events"].tolist()))["care"]
                            for s in g.seed.unique()])
                   for _, g in grid.groupby(["q", "steps"]))
        choice = sel.drop_duplicates("heldout")[["heldout", "q", "steps"]]
        out.append(dict(method=method, variant=var, seeds=len(keys),
                        care_fixed=np.mean([c["care"] for c in fixed]),
                        care_lofo=np.mean([c["care"] for c in cares]),
                        care_lofo_sd=np.std([c["care"] for c in cares]),
                        detected_lofo=np.mean([c["detected"] for c in cares]),
                        false_alarms_lofo=np.mean([c["false_alarms"] for c in cares]),
                        care_insample_best=best,
                        choices="; ".join(f"{r.heldout}: q={r.q}, k={r.steps}"
                                          for r in choice.itertuples())))
    return pd.DataFrame(out)


def bootstrap_table(raw, pairs, n_boot=1000, seed=0):
    """Paired, label-stratified bootstrap over datasets of CARE differences (seed-averaged).
    Each pair is compared on the datasets both configurations finished."""
    per = {}

    def rows_of(cfg):
        if cfg not in per:
            m, v = cfg.split("|")
            per[cfg] = [per_event_setting(raw, k).set_index(["farm", "event_id"])
                        for k in keys_of(raw, m, v)]
        return per[cfg]

    def care(rows, ids):
        return np.mean([L.care_score(r.loc[ids])["care"] for r in rows])

    rng = np.random.default_rng(seed)
    out = []
    for a, b in pairs:
        ra, rb = rows_of(a), rows_of(b)
        if not ra or not rb:
            continue
        common = ra[0].index
        for r in ra[1:] + rb:
            common = common.intersection(r.index)
        common = common.sort_values()
        lab = ra[0].loc[common, "label"].to_numpy()
        an, no = common[lab == "anomaly"], common[lab == "normal"]
        if len(an) == 0 or len(no) == 0:
            continue
        d = []
        for _ in range(n_boot):
            ids = an[rng.integers(0, len(an), len(an))].append(no[rng.integers(0, len(no), len(no))])
            d.append(care(ra, ids) - care(rb, ids))
        d = np.array(d)
        out.append(dict(a=a, b=b, n_datasets=len(common), care_a=care(ra, common),
                        care_b=care(rb, common), diff=care(ra, common) - care(rb, common),
                        ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                        share_le0=float((d <= 0).mean())))
    return pd.DataFrame(out)


PAIRS = [("ae|smooth", "ae|raw"), ("ae|cn", "ae|raw"), ("ae|cn_smooth", "ae|smooth"),
         ("ae|cn_smooth", "ae|raw"), ("cae|raw", "ae|raw"), ("cae|smooth", "ae|cn_smooth"),
         ("ae|smooth", "pca|smooth"), ("ae|cn_smooth", "pca|cn_smooth"),
         ("iforest|cn_smooth", "iforest|raw"), ("ae|raw", "ae_sigmoid|raw"),
         ("lstm|raw", "ae|raw"), ("lstm|smooth", "ae|smooth"), ("lstm|cn_smooth", "ae|cn_smooth")]


def robustness_table(raw):
    out = []
    keys = sorted({k for _, d in raw for k in d if k.startswith("rob:")})
    for k in keys:
        _, corr, key = k.split(":")
        rows = []
        for meta, data in raw:
            if k in data:
                r = L.care_event(data[k] > meta["thresholds"][key], data["eval_normal"],
                                 data["in_window"], meta["label"])
                rows.append(dict(event_id=meta["event_id"], farm=meta["farm"], **r))
        c = L.care_score(pd.DataFrame(rows))
        out.append(dict(corruption=corr, key=key, **c))
    for key in sorted({k.split(":")[2] for k in keys}):     # clean reference
        c = L.care_score(per_event_setting(raw, key))
        out.append(dict(corruption="none", key=key, **c))
    return pd.DataFrame(out).sort_values(["key", "corruption"]) if out else pd.DataFrame()


def sigmoid_table(logs: pd.DataFrame) -> pd.DataFrame:
    """Best validation loss per dataset: linear vs sigmoid head (seed 0)."""
    if logs.empty:
        return logs
    best = logs.groupby(["event_id", "base"])["val"].min().unstack()
    cols = [c for c in ("ae_s0", "ae_sigmoid_s0", "cae_s0", "lstm_s0") if c in best]
    return best[cols].describe().T.reset_index().rename(columns={"index": "base"})


def latex_care(t: pd.DataFrame) -> str:
    """Main table: every detector x post-processing, all 95 datasets."""
    t = t[t["farm"] == "all"].copy()
    order = {m: i for i, m in enumerate(BASE_LABEL)}
    t["o"] = t["method"].map(order).fillna(99)
    t = t.sort_values(["o", "variant"])
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Detector & Post-processing & CARE & Cov. & Acc. & Rel. & Earl.\\", r"\midrule"]
    for _, r in t.iterrows():
        sd = f"$\\pm${r['care_sd']:.3f}" if r["seeds"] > 1 else ""
        lines.append(f"{BASE_LABEL.get(r['method'], r['method'])} & {VAR_LABEL[r['variant']]} & "
                     f"{r['care']:.3f}{sd} & {r['coverage']:.2f} & {r['accuracy']:.2f} & "
                     f"{r['reliability']:.2f} & {r['earliness']:.2f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="paper_tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = load_raw(a.inputs)
    ev = load_events(raw)
    ev.to_csv(os.path.join(a.out, "per_event.csv"), index=False)
    n = ev.drop_duplicates("event_id")
    print(f"{len(n)} datasets merged: " + ", ".join(
        f"farm {f}: {(g['label'] == 'anomaly').sum()} anomaly / {(g['label'] == 'normal').sum()} normal"
        for f, g in n.groupby("farm")))
    care = care_table(ev)
    care.to_csv(os.path.join(a.out, "care.csv"), index=False)
    with open(os.path.join(a.out, "care.tex"), "w") as f:
        f.write(latex_care(care))
    show = ["method", "variant", "farm", "care", "care_sd", "coverage", "accuracy", "reliability",
            "earliness", "detected", "n_anomaly", "false_alarms", "n_normal"]
    print("\n== CARE score ==\n" + care[care.farm == "all"][show].round(3).to_string(index=False))

    chans = [f for d in a.inputs for f in glob.glob(os.path.join(d, "channels.csv"))]
    if chans:
        att = attribution_table(pd.concat([pd.read_csv(f) for f in chans]))
        att.to_csv(os.path.join(a.out, "attribution.csv"), index=False)
        att.round(3).to_latex(os.path.join(a.out, "attribution.tex"), index=False, na_rep="--")
        print("\n== attribution ==\n" + att.round(3).to_string(index=False))

    steps = []
    for m, v in MAIN_CONFIGS:
        for st in L.TUNE_STEPS:
            cs = [L.care_score(per_event_setting(raw, k, steps=st)) for k in keys_of(raw, m, v)]
            if cs:
                steps.append(dict(method=m, variant=v, steps=st,
                                  **{x: np.mean([c[x] for c in cs])
                                     for x in ("care", "detected", "false_alarms")}))
    pd.DataFrame(steps).to_csv(os.path.join(a.out, "counter_sensitivity.csv"), index=False)

    for name, fn in (("bootstrap", lambda: bootstrap_table(raw, PAIRS)),
                     ("lofo", lambda: lofo_table(raw)),
                     ("robustness", lambda: robustness_table(raw))):
        try:                       # one failing table must not cost the others
            t = fn()
        except Exception:
            import traceback
            traceback.print_exc()
            continue
        t.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
        print(f"\n== {name} ==\n" + t.round(3).to_string(index=False))

    tim = [f for d in a.inputs for f in glob.glob(os.path.join(d, "timing.csv"))]
    if tim:
        t = pd.concat([pd.read_csv(f) for f in tim])
        t = t.groupby(["farm", "detector"])[["fit_s", "score_s_per_1k"]].median().reset_index()
        t.to_csv(os.path.join(a.out, "timing.csv"), index=False)
        print("\n== median compute time per dataset ==\n" + t.round(3).to_string(index=False))

    logs = [f for d in a.inputs for f in glob.glob(os.path.join(d, "trainlog.csv"))]
    if logs:
        t = sigmoid_table(pd.concat([pd.read_csv(f) for f in logs]))
        t.to_csv(os.path.join(a.out, "val_loss_heads.csv"), index=False)
        print("\n== best validation loss per dataset ==\n" + t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
