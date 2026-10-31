"""Merge the three shard outputs into the paper tables.

    python merge_results.py --inputs shard0_dir shard1_dir shard2_dir --out paper_tables
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import wtpm_lib as L

ORDER = ["ae_linear", "ae_sigmoid", "isolation_forest", "pca", "three_sigma"]


def load(inputs, stem):
    files = sorted(f for d in inputs for f in glob.glob(os.path.join(d, f"{stem}_shard*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def detection_table(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in ORDER:
        r = res[res["method"] == m]
        if r.empty:
            continue
        per_seed = []
        for seed, g in r.groupby("seed"):
            an, no = g[g["label"] == "anomaly"], g[g["label"] == "normal"]
            tp, fp, fn, tn = (g[c].sum() for c in ("tp", "fp", "fn", "tn"))
            det = int(an["detected"].astype(bool).sum())
            fa = int(no["alarm_any"].astype(bool).sum())
            prec = tp / (tp + fp) if tp + fp else np.nan
            rec = tp / (tp + fn) if tp + fn else np.nan
            prev = (an["tp"] + an["fn"]) / (an["tp"] + an["fn"] + an["fp"] + an["tn"])
            per_seed.append(dict(
                events_detected=det, n_anomaly=len(an), false_alarm_events=fa, n_normal=len(no),
                precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else np.nan,
                fpr=fp / (fp + tn) if fp + tn else np.nan,
                roc_auc=an["roc_auc"].mean(), pr_auc=an["pr_auc"].mean(),
                pr_auc_prevalence=prev.mean(),
                median_first_alarm_h=an.loc[an["detected"].astype(bool),
                                            "first_alarm_h_rel_start"].median()))
        p = pd.DataFrame(per_seed)
        row = {"method": m, "seeds": len(p)}
        for c in p.columns:
            row[c] = p[c].mean()
            if len(p) > 1 and c not in ("n_anomaly", "n_normal"):
                row[c + "_sd"] = p[c].std()
        lo, hi = L.wilson(int(round(row["events_detected"])), int(row["n_anomaly"]))
        row["detect_rate_ci"] = f"[{lo:.2f}, {hi:.2f}]"
        lo, hi = L.wilson(int(round(row["false_alarm_events"])), int(row["n_normal"]))
        row["false_alarm_rate_ci"] = f"[{lo:.2f}, {hi:.2f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def attribution_table(rca: pd.DataFrame) -> pd.DataFrame:
    """Top-1 / top-3 hit rate against the hand-labelled subsystem of each anomaly event."""
    labelled = {e: s for e, s in L.EVENT_SUBSYSTEMS.items() if s}
    rca = rca[rca["event_id"].isin(labelled)]
    rows = []
    for agg in ("mean_ratio", "sum_ratio"):
        per_seed = []
        for seed, g in rca.groupby("seed"):
            h1 = h3 = n = 0
            for eid, e in g.groupby("event_id"):
                ranked = e.sort_values(agg, ascending=False)["subsystem"].tolist()
                truth = set(labelled[eid])
                h1 += bool(truth & set(ranked[:1]))
                h3 += bool(truth & set(ranked[:3]))
                n += 1
            per_seed.append((h1 / n, h3 / n, n))
        a = np.array(per_seed)
        rows.append(dict(rule=f"AE residual, {agg.split('_')[0]} over channels", top1=a[:, 0].mean(),
                         top1_sd=a[:, 0].std(), top3=a[:, 1].mean(), top3_sd=a[:, 1].std(),
                         n_events=int(a[0, 2])))
    # floors a reviewer will ask for: always guess the commonest label, and random ranking
    n_sub = rca["subsystem"].nunique()
    events = rca["event_id"].unique()
    counts = pd.Series([s for e in events for s in labelled[e]]).value_counts()
    top_guess = counts.index[:3].tolist()
    rows.append(dict(rule=f"always '{top_guess[0]}' / top-3 commonest",
                     top1=np.mean([top_guess[0] in labelled[e] for e in events]),
                     top3=np.mean([bool(set(top_guess) & set(labelled[e])) for e in events]),
                     n_events=len(events)))
    rows.append(dict(rule="random ranking (expected)",
                     top1=np.mean([len(labelled[e]) / n_sub for e in events]),
                     top3=np.mean([1 - np.prod([(n_sub - len(labelled[e]) - i) / (n_sub - i)
                                                for i in range(3)]) for e in events]),
                     n_events=len(events)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="paper_tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = load(a.inputs, "results")
    rca = load(a.inputs, "rca")
    print(f"{res['event_id'].nunique()} datasets merged "
          f"({(res.drop_duplicates('event_id')['label'] == 'anomaly').sum()} anomaly)")
    det = detection_table(res)
    att = attribution_table(rca)
    for name, t in (("detection", det), ("attribution", att)):
        t.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
        t.round(3).to_latex(os.path.join(a.out, f"{name}.tex"), index=False, na_rep="--")
        print(f"\n== {name} ==\n{t.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
