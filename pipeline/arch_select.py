"""Run D analysis: does choosing the AE architecture per farm close the gap to the benchmark?

Six AE configurations (3 sizes x 2 learning rates, seed 0, see wtpm_lib.ARCH_CONFIGS) were
trained on every dataset. They are compared under these strategies, all fixed in advance:

    fixed <cfg>        one configuration for every farm (arch_m_hi is the paper's AE)
    val per farm       per farm, the configuration with the lowest mean rank of best
                       validation loss over that farm's datasets (no fault labels used)
    val per dataset    per dataset, the configuration with the lowest best validation loss
    lofo config        configuration chosen by CARE on the two other farms (threshold fixed)
    lofo joint         configuration, threshold quantile and counter chosen on the two other
                       farms, applied unchanged to the held-out farm
    oracle per farm    configuration (and, for 'oracle joint', q and k) chosen by CARE on the
                       farm itself: an optimistic ceiling that uses the test labels, not a result

    python pipeline/arch_select.py --inputs out/benchmark --out tables
"""
import argparse
import os

import numpy as np
import pandas as pd

import merge_results as M
import wtpm_lib as L

VARIANTS = ("raw", "cn_smooth")


def best_val(trainlog: pd.DataFrame) -> pd.DataFrame:
    """Best validation loss per dataset and configuration."""
    t = trainlog.copy()
    t["config"] = t["base"].str.replace(r"_s\d+$", "", regex=True)
    return t.groupby(["event_id", "config"])["val"].min().rename("best_val").reset_index()


def choose_by_val(bv: pd.DataFrame, farm_of: dict) -> tuple[dict, dict]:
    """Label-free choices: per farm (lowest mean within-dataset rank) and per dataset."""
    bv = bv.copy()
    bv["rank"] = bv.groupby("event_id")["best_val"].rank(method="min")
    bv["farm"] = bv["event_id"].map(farm_of)
    per_farm = (bv.groupby(["farm", "config"])["rank"].mean().reset_index()
                .sort_values(["farm", "rank", "config"]).drop_duplicates("farm")
                .set_index("farm")["config"].to_dict())
    per_ds = (bv.sort_values(["event_id", "best_val", "config"]).drop_duplicates("event_id")
              .set_index("event_id")["config"].to_dict())
    return per_farm, per_ds


def assemble(rows: dict, pick) -> pd.DataFrame:
    """Per-event rows where pick(farm, event_id) names the setting used for that dataset."""
    out = []
    for key, ev in rows.items():
        for _, r in ev.iterrows():
            if pick(r["farm"], r["event_id"]) == key:
                out.append(r)
    return pd.DataFrame(out)


def _care(ev: pd.DataFrame) -> float:
    """CARE score, with undefined values (no rows, or no normal dataset) ranked last."""
    if len(ev) == 0:
        return -np.inf
    c = L.care_score(ev)["care"]
    return c if np.isfinite(c) else -np.inf


def lofo_pick(rows: dict, farms) -> dict:
    """Per held-out farm, the setting with the best CARE on the other farms."""
    choice = {}
    for f in farms:
        ranked = sorted(rows.items())
        best, best_c = ranked[0][0], -np.inf
        for key, ev in ranked:
            c = _care(ev[ev["farm"] != f])
            if c > best_c + 1e-12:
                best, best_c = key, c
        choice[f] = best
    return choice


def oracle_pick(rows: dict, farms) -> dict:
    """Per farm, the setting with the best CARE on that farm itself (uses test labels)."""
    choice = {}
    for f in farms:
        ranked = sorted(rows.items())
        best, best_c = ranked[0][0], -np.inf
        for key, ev in ranked:
            c = _care(ev[ev["farm"] == f])
            if c > best_c + 1e-12:
                best, best_c = key, c
        choice[f] = best
    return choice


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot=1000, seed=0) -> dict:
    """Label-stratified paired bootstrap of CARE(a) - CARE(b) over common datasets."""
    a = a.set_index(["farm", "event_id"]).sort_index()
    b = b.set_index(["farm", "event_id"]).sort_index()
    common = a.index.intersection(b.index)
    lab = a.loc[common, "label"].to_numpy()
    an, no = common[lab == "anomaly"], common[lab == "normal"]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ids = an[rng.integers(0, len(an), len(an))].append(no[rng.integers(0, len(no), len(no))])
        d.append(L.care_score(a.loc[ids])["care"] - L.care_score(b.loc[ids])["care"])
    diff = L.care_score(a.loc[common])["care"] - L.care_score(b.loc[common])["care"]
    return dict(diff=diff, ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)))


def summarise(name, variant, ev, choice="") -> dict:
    if len(ev) == 0:
        return dict(strategy=name, variant=variant, n=0, choice=choice)
    c = L.care_score(ev)
    return dict(strategy=name, variant=variant, n=len(ev), choice=choice,
                **{k: c[k] for k in ("care", "coverage", "accuracy", "reliability", "earliness",
                                     "detected", "false_alarms")})


def analyse(raw, trainlog: pd.DataFrame):
    farm_of = {m["event_id"]: m["farm"] for m, _ in raw}
    farms = sorted(set(farm_of.values()))
    bv = best_val(trainlog)
    vf, vd = choose_by_val(bv, farm_of)
    rows_out, boot = [], []
    for var in VARIANTS:
        fixed = {cfg: M.per_event_setting(raw, f"{cfg}_s0|{var}") for cfg in L.ARCH_CONFIGS}
        fixed = {k: v for k, v in fixed.items() if len(v)}
        if not fixed:
            continue
        for cfg, ev in fixed.items():
            rows_out.append(summarise(f"fixed {cfg}", var, ev))
        by_val_farm = assemble(fixed, lambda f, e: vf.get(f))
        by_val_ds = assemble(fixed, lambda f, e: vd.get(e))
        lofo = lofo_pick(fixed, farms)
        by_lofo = assemble(fixed, lambda f, e: lofo.get(f))
        orc = oracle_pick(fixed, farms)
        by_orc = assemble(fixed, lambda f, e: orc.get(f))
        rows_out += [summarise("val per farm", var, by_val_farm, str(vf)),
                     summarise("val per dataset", var, by_val_ds,
                               str(pd.Series(vd).value_counts().to_dict())),
                     summarise("lofo config", var, by_lofo, str(lofo)),
                     summarise("oracle per farm (ceiling)", var, by_orc, str(orc))]
        # joint search over configuration, quantile and counter
        joint = {}
        for cfg in fixed:
            for q in L.TUNE_Q:
                for st in L.TUNE_STEPS:
                    joint[(cfg, q, st)] = M.per_event_setting(raw, f"{cfg}_s0|{var}", q, st)
        jl = lofo_pick(joint, farms)
        jo = oracle_pick(joint, farms)
        rows_out += [summarise("lofo joint", var, assemble(joint, lambda f, e: jl.get(f)), str(jl)),
                     summarise("oracle joint (ceiling)", var,
                               assemble(joint, lambda f, e: jo.get(f)), str(jo))]
        ref = fixed.get(L.ARCH_REFERENCE)
        if ref is not None:
            for name, ev in (("val per farm", by_val_farm), ("val per dataset", by_val_ds),
                             ("lofo config", by_lofo),
                             ("lofo joint", assemble(joint, lambda f, e: jl.get(f)))):
                boot.append(dict(a=name, b=f"fixed {L.ARCH_REFERENCE}", variant=var,
                                 **paired_bootstrap(ev, ref)))
    val = bv.assign(farm=bv["event_id"].map(farm_of)).groupby(["farm", "config"])["best_val"] \
        .median().unstack()
    return pd.DataFrame(rows_out), pd.DataFrame(boot), val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = M.load_raw(a.inputs)
    logs = pd.concat([pd.read_csv(os.path.join(d, "trainlog.csv")) for d in a.inputs
                      if os.path.exists(os.path.join(d, "trainlog.csv"))])
    res, boot, val = analyse(raw, logs)
    res.to_csv(os.path.join(a.out, "arch_search.csv"), index=False)
    boot.to_csv(os.path.join(a.out, "arch_bootstrap.csv"), index=False)
    val.to_csv(os.path.join(a.out, "arch_val_loss.csv"))
    pd.set_option("display.width", 220)
    print("== architecture search ==\n" + res.drop(columns="choice").round(3).to_string(index=False))
    print("\n== choices ==\n" + res[["strategy", "variant", "choice"]].to_string(index=False))
    print("\n== bootstrap against the paper's AE ==\n" + boot.round(3).to_string(index=False))
    print("\n== median best validation loss ==\n" + val.round(4).to_string())


if __name__ == "__main__":
    main()
