"""Turn the stored scores into the paper tables (CARE score, detection, attribution).

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
              "iforest": "Isolation Forest", "pca": "PCA", "three_sigma": "3-sigma fraction"}
VAR_LABEL = {"raw": "raw", "smooth": "+ smoothing", "cn": "+ condition norm.",
             "cn_smooth": "+ cond. norm. + smoothing"}


def load_events(inputs):
    rows = []
    for d in inputs:
        for js in sorted(glob.glob(os.path.join(d, "scores", "*.json"))):
            npz = js[:-5] + ".npz"
            if not os.path.exists(npz):
                continue
            meta = json.load(open(js))
            data = np.load(npz)
            for key, thr in meta["thresholds"].items():
                base, var = key.split("|")
                r = L.care_event(data[key] > thr, data["eval_normal"], data["in_window"],
                                 meta["label"])
                group, seed = (base.rsplit("_s", 1) + ["0"])[:2] if base.startswith("ae") \
                    else (base, "0")
                rows.append(dict(event_id=meta["event_id"], farm=meta["farm"], method=group,
                                 seed=int(seed), variant=var, **r))
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
    ev = load_events(a.inputs)
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
    print("\n== CARE score ==\n" + care[show].round(3).to_string(index=False))

    chans = [f for d in a.inputs for f in glob.glob(os.path.join(d, "channels.csv"))]
    if chans:
        att = attribution_table(pd.concat([pd.read_csv(f) for f in chans]))
        att.to_csv(os.path.join(a.out, "attribution.csv"), index=False)
        att.round(3).to_latex(os.path.join(a.out, "attribution.tex"), index=False, na_rep="--")
        print("\n== attribution ==\n" + att.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
