"""Case study of one dataset: health index, time-to-threshold with an uncertainty band,
maintenance-gate sensitivity and subsystem attribution. For farm C datasets also a
temperature normal-behaviour model (tested on another turbine) and the Unity export.

    python analyze_case.py --farm C --event 44 --cross 61 --out out/case_44
    python analyze_case.py --farm A --event 40 --out out/case_40
CPU is enough (one autoencoder). Robustness to noise / missing data is now part of the
benchmark (all datasets), not of the case study.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wtpm_lib as L

HI_TH = 0.20          # failure floor for the health index (planning constant)
SLOPE_WIN = 144       # 24 h causal window for the HI slope
TTT_CAP = 720.0       # hours; anything slower is shown as "steady"
GATES = (72, 168, 336)
BUFFERS = (12, 24, 48)
TEMP_INPUTS = ["sensor_144_avg", "power_6_avg", "sensor_7_avg", "wind_speed_236_avg"]
TEMP_TARGETS = {"rotor_bearing_temp": "sensor_196_avg", "cooling_water_outlet": "sensor_233_avg"}
UNITY_COLS = {"wind_speed_ms": "wind_speed_236_avg", "rotor_speed_rpm": "sensor_144_avg",
              "power_kw": "power_6_avg", "gearbox_oil_temp_c": "sensor_186_avg",
              "rotor_bearing_temp_c": "sensor_196_avg", "cooling_water_out_c": "sensor_233_avg"}


def health_index(score, cal_idx):
    """1 at the typical calibration score, 0.5 at the alarm threshold, towards 0 above it.
    `score` is the already smoothed, condition-normalised score (causal)."""
    q50, q99 = np.median(score[cal_idx]), np.quantile(score[cal_idx], L.THRESHOLD_Q)
    return 1.0 / (1.0 + np.maximum(0.0, score - q50) / (q99 - q50)), (q50, q99)


def time_to_threshold(hi, hours, win=SLOPE_WIN):
    """Causal rolling OLS of HI on time. Returns point estimate and 95 % band (hours)."""
    t, y = pd.Series(hours), pd.Series(hi)
    n = t.rolling(win).count()
    st, sy = t.rolling(win).sum(), y.rolling(win).sum()
    stt, sty, syy = (t * t).rolling(win).sum(), (t * y).rolling(win).sum(), (y * y).rolling(win).sum()
    sxx = stt - st * st / n
    sxy = sty - st * sy / n
    b = sxy / sxx
    rss = (syy - sy * sy / n) - b * sxy
    se = np.sqrt(np.maximum(rss, 0) / (n - 2) / sxx)

    def ttt(slope):
        out = np.where(slope < 0, (hi - HI_TH) / -np.minimum(slope, -1e-12), np.inf)
        out = np.where(hi <= HI_TH, 0.0, out)
        return np.minimum(out, TTT_CAP)

    b, se = b.to_numpy(), se.to_numpy()
    return ttt(b), ttt(b - 1.96 * se), ttt(b + 1.96 * se), b


def block_bootstrap(y, p, days, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ud = np.unique(days)
    idx = {d: np.flatnonzero(days == d) for d in ud}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[d] for d in rng.choice(ud, len(ud))])
        vals.append(fn(y[pick], p[pick]))
    return np.percentile(vals, [2.5, 97.5])


def temp_metrics(y, p, days):
    from sklearn.metrics import mean_absolute_error, r2_score
    f = {"MAE": mean_absolute_error,
         "RMSE": lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))),
         "R2": r2_score,
         "MAPE": lambda a, b: float(np.mean(np.abs((a - b) / np.where(a == 0, np.nan, a)))) * 100}
    out = {}
    for k, fn in f.items():
        lo, hi = block_bootstrap(y, p, days, fn)
        out[k] = (float(fn(y, p)), float(lo), float(hi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="out_44")
    ap.add_argument("--farm", default="C")
    ap.add_argument("--event", type=int, default=44)
    ap.add_argument("--cross", type=int, default=61, help="farm C only; 0 = skip")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    events = L.load_event_info(a.data_dir, a.farm).set_index("event_id")
    feats = L.load_features(a.data_dir, a.farm)
    ev = events.loc[a.event].copy()
    ev["event_id"] = a.event
    df = L.load_dataset(a.event, a.data_dir, a.farm)
    prep = L.prepare(df, feats, farm=a.farm, train_filter=True)
    tag = str(a.event)
    farm_c = a.farm == "C" and a.cross > 0
    errfn, _ = L.train_autoencoder(prep.X[prep.fit_idx], prep.X[prep.val_idx], seed=0,
                                   epochs=a.epochs)
    ch = errfn(prep.X)
    score = L.variants(ch.mean(axis=1), prep)["cn_smooth"]     # the full pipeline's score
    thr = float(np.quantile(score[prep.cal_b], L.THRESHOLD_Q))
    summary = {"event": a.event, "farm": a.farm, "description": ev["event_description"], "threshold": thr,
               "n_features": len(prep.columns),
               "detection": L.evaluate(score, prep, ev, thr)}

    # health index, alarms, time to threshold (all causal)
    ts = df["time_stamp"]
    hours = ((ts - ts.iloc[0]).dt.total_seconds() / 3600).to_numpy()
    hi, (q50, q99) = health_index(score, prep.cal_b)
    crit = L.criticality(score > thr, prep.eval_normal)
    alarm = crit >= L.ALARM_STEPS
    ttt, ttt_lo, ttt_hi, slope = time_to_threshold(hi, hours)
    start, end = ev["event_start"], ev["event_end"]
    in_ev = ((ts >= start) & (ts <= end)).to_numpy()
    pm = prep.pred_mask

    # maintenance gate sensitivity: when would a work order first be raised?
    gate_rows = []
    pre_event = prep.pred_mask & (ts < start).to_numpy()
    for g in GATES:
        trig = alarm & (ttt <= g)
        for buf in BUFFERS:
            first = np.flatnonzero(trig & prep.pred_mask)
            first_h = (ts.iloc[first[0]] - start).total_seconds() / 3600 if len(first) else np.nan
            visit_h = first_h + max(ttt[first[0]] - buf, 0) if len(first) else np.nan
            gate_rows.append(dict(gate_h=g, buffer_h=buf, first_trigger_h_rel_start=first_h,
                                  visit_h_rel_start=visit_h,
                                  triggers_in_validation=int(trig[prep.cal_b].sum()),
                                  triggers_before_event=int((trig & pre_event).sum())))
    pd.DataFrame(gate_rows).to_csv(os.path.join(a.out, f"gate_sensitivity_{tag}.csv"), index=False)

    # attribution on the labelled window
    col_sub = L.column_subsystems(prep.columns, feats)
    rca = L.subsystem_scores(ch, prep, ev, col_sub).sort_values("mean_ratio", ascending=False)
    rca.to_csv(os.path.join(a.out, f"rca_{tag}.csv"), index=False)
    truth = L.EVENT_SUBSYSTEMS.get(a.event) or []
    summary["rca_rank_of_truth"] = {t: int(rca["subsystem"].tolist().index(t)) + 1
                                    for t in truth if t in rca["subsystem"].tolist()}

    preds_fault = {}
    if farm_c:
        # temperature normal-behaviour model: train on 44 fit rows, test on 44 validation rows
        # (temporal hold-out), on another turbine's training year (cross-turbine), and on the fault
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        cross = L.load_dataset(a.cross, a.data_dir)
        cross_rows = (cross["train_test"] == "train") & (cross["status_type_id"] == 0)
        temp_rows = []
        preds_fault = {}
        for name, tgt in TEMP_TARGETS.items():
            cols = TEMP_INPUTS + [tgt]
            d44 = df[cols].apply(pd.to_numeric, errors="coerce")
            ok = d44.notna().all(axis=1).to_numpy() & (df["status_type_id"] == 0).to_numpy()
            fit = np.intersect1d(prep.fit_idx, np.flatnonzero(ok))
            val = np.intersect1d(prep.val_idx, np.flatnonzero(ok))
            Xtr, ytr = d44.iloc[fit][TEMP_INPUTS].to_numpy(), d44.iloc[fit][tgt].to_numpy()
            dc = cross.loc[cross_rows, cols].apply(pd.to_numeric, errors="coerce").dropna()
            sets = {"temporal_holdout_44": (d44.iloc[val], ts.iloc[val]),
                    f"cross_turbine_{a.cross}": (dc, cross.loc[dc.index, "time_stamp"])}
            for mname, model in (("dummy_mean", DummyRegressor()), ("linear", LinearRegression()),
                                 ("hist_gbm", HistGradientBoostingRegressor(random_state=0))):
                model.fit(Xtr, ytr)
                for sname, (d, t) in sets.items():
                    p = model.predict(d[TEMP_INPUTS].to_numpy())
                    m = temp_metrics(d[tgt].to_numpy(), p, t.dt.floor("D").to_numpy())
                    temp_rows.append(dict(target=name, model=mname, test=sname,
                                          **{f"{k}": v[0] for k, v in m.items()},
                                          **{f"{k}_lo": v[1] for k, v in m.items()},
                                          **{f"{k}_hi": v[2] for k, v in m.items()}))
                if mname == "hist_gbm":
                    pf = np.full(len(df), np.nan)
                    rows = np.flatnonzero(d44.notna().all(axis=1).to_numpy())
                    pf[rows] = model.predict(d44.iloc[rows][TEMP_INPUTS].to_numpy())
                    preds_fault[name] = pf
                    res = d44[tgt].to_numpy() - pf
                    summary[f"{name}_residual_mean_val"] = float(np.nanmean(res[val]))
                    summary[f"{name}_residual_mean_event"] = float(np.nanmean(res[in_ev & ok]))
        pd.DataFrame(temp_rows).to_csv(os.path.join(a.out, f"temperature_model_{tag}.csv"), index=False)

    if farm_c:
        # Unity export: measured SCADA values plus model outputs, nothing synthetic
        unity = pd.DataFrame({"time_stamp": ts})
        for k, c in UNITY_COLS.items():
            unity[k] = pd.to_numeric(df[c], errors="coerce")
        top = rca["subsystem"].iloc[0] if len(rca) else "NONE"
        unity = unity.assign(anomaly_score=score, alarm=alarm, health_index=hi, ttt_h=ttt,
                             ttt_lo_h=ttt_lo, ttt_hi_h=ttt_hi,
                             top_subsystem=np.where(alarm, top, ""))
        unity.to_csv(os.path.join(a.out, f"unity_telemetry_{tag}.csv"), index=False)

    # figures
    def shade(ax):
        ax.axvspan(start, end, color="tab:red", alpha=0.12, label="labelled event")
        ax.axvline(ts[pm].iloc[0], color="k", ls=":", lw=1, label="prediction starts")

    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(ts, hi, lw=0.8, label="health index")
    ax.fill_between(ts, 0, 1, where=alarm, color="tab:orange", alpha=0.3, step="mid",
                    label="alarm (12 h counter)")
    shade(ax)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("HI")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"fig_hi_{tag}.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(7, 2.6))
    w = pm
    ax.plot(ts[w], ttt[w], lw=0.8, label="time to HI=0.2")
    ax.fill_between(ts[w], ttt_lo[w], ttt_hi[w], alpha=0.25, label="95% band (slope SE)")
    ax.axhline(168, color="k", ls="--", lw=0.8, label="168 h gate")
    shade(ax)
    ax.set_ylabel("hours")
    ax.set_ylim(0, TTT_CAP)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"fig_ttt_{tag}.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.barh(rca["subsystem"][::-1], rca["mean_ratio"][::-1])
    ax.set_xlabel("event error / validation error")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"fig_rca_{tag}.png"), dpi=200)

    if farm_c:
        name = "cooling_water_outlet"
        fig, ax = plt.subplots(figsize=(7, 2.6))
        meas = pd.to_numeric(df[TEMP_TARGETS[name]], errors="coerce")
        ax.plot(ts[w], meas[w], lw=0.6, label="measured")
        ax.plot(ts[w], preds_fault[name][w], lw=0.6, label="expected (normal-behaviour model)")
        shade(ax)
        ax.set_ylabel("cooling water out (°C)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"fig_temp_{tag}.png"), dpi=200)

    with open(os.path.join(a.out, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
