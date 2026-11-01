import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import wtpm_lib as L  # noqa: E402
from analyze_case import time_to_threshold  # noqa: E402


def test_criticality_counts_up_down_and_freezes_on_abnormal_status():
    anom = np.array([1, 1, 1, 0, 1, 0, 0, 0], bool)
    norm = np.array([1, 1, 0, 1, 1, 1, 1, 1], bool)
    assert L.criticality(anom, norm).tolist() == [1, 2, 2, 1, 2, 1, 0, 0]


@pytest.mark.parametrize("desc,expected", [
    ("Cooling water temp. generator outlet 1", "COOLING"),
    ("Generator cooling air temperature A", "COOLING"),
    ("Axis 1 cooling element temperature", "PITCH_HUB"),
    ("Hydraulic pressure azimuth circuit A", "YAW"),
    ("Hydraulic pressure rotor brake A", "HYDRAULIC_BRAKE"),
    ("Motor current gear oil pump A", "GEARBOX"),
    ("Electrical cabinet ambient temperature", "ELECTRICAL"),
    ("Ambient temperature", "ENVIRONMENT"),
    ("Rotor bearing temperature 1", "ROTOR_DRIVETRAIN"),
    ("Something unseen", "OTHER"),
])
def test_subsystem_rules(desc, expected):
    assert L.subsystem_of(desc) == expected


def test_base_sensor_strips_statistic_suffix():
    assert L.base_sensor("wind_speed_236_std") == "wind_speed_236"
    assert L.base_sensor("sensor_12_avg") == "sensor_12"


def test_wilson_interval_contains_point_estimate():
    lo, hi = L.wilson(13, 26)
    assert lo < 0.5 < hi


def _toy_prepared():
    n = 400
    ids = np.arange(n)
    df = pd.DataFrame({
        "time_stamp": pd.date_range("2023-01-01", periods=n, freq="10min"),
        "id": ids, "train_test": np.where(ids < 200, "train", "prediction"),
        "status_type_id": 0})
    pred = ids >= 200
    return L.Prepared(np.zeros((n, 1), np.float32), ["x"], ids[:150], ids[150:200], pred,
                      np.ones(n, bool), df)


def test_evaluate_detects_sustained_event_and_not_before_it():
    prep = _toy_prepared()
    ev = pd.Series(dict(event_id=1, event_label="anomaly", event_start_id=300, event_end_id=399,
                        event_start=prep.df["time_stamp"][300]))
    score = np.zeros(400)
    score[300:] = 5.0
    r = L.evaluate(score, prep, ev, threshold=1.0)
    assert r["detected"] and r["tp"] == 100 and r["fp"] == 0
    assert r["first_alarm_h_rel_start"] == pytest.approx(71 / 6)   # 72nd flagged row
    assert r["roc_auc"] == 1.0


def test_evaluate_normal_event_counts_false_alarm():
    prep = _toy_prepared()
    ev = pd.Series(dict(event_id=2, event_label="normal", event_start_id=300, event_end_id=399,
                        event_start=prep.df["time_stamp"][300]))
    score = np.zeros(400)
    score[220:300] = 5.0
    r = L.evaluate(score, prep, ev, threshold=1.0)
    assert r["alarm_any"] and np.isnan(r["detected"]) and r["fp"] == 80


def test_time_to_threshold_on_a_linear_decline():
    hours = np.arange(300) / 6.0
    hi = 1.0 - 0.01 * hours                       # loses 0.01 per hour
    ttt, lo, hi_band, slope = time_to_threshold(hi, hours, win=30)
    assert slope[-1] == pytest.approx(-0.01)
    assert ttt[-1] == pytest.approx((hi[-1] - 0.2) / 0.01)
    assert lo[-1] <= ttt[-1] <= hi_band[-1]


def test_fetch_member_retries_and_never_leaves_partial_file(tmp_path, monkeypatch):
    import types
    calls = {"n": 0}

    class FakeZip:
        def __init__(self, url, **kw):
            assert "timeout" in kw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract(self, member, dest):
            calls["n"] += 1
            path = os.path.join(dest, member)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write("partial" if calls["n"] == 1 else "a;b\n1;2\n")
            if calls["n"] == 1:
                raise ConnectionError("dropped")

    monkeypatch.setitem(sys.modules, "remotezip", types.SimpleNamespace(RemoteZip=FakeZip))
    out = L.fetch_member("x/y.csv", str(tmp_path), wait_s=0)
    assert calls["n"] == 2
    assert open(out).read() == "a;b\n1;2\n"
    assert sorted(os.listdir(tmp_path)) == ["x"]          # temp folders cleaned up


# ----------------------------------------------------------------- v2 additions

@pytest.mark.parametrize("desc,expected", [
    ("Tower vibration axis Y", "NACELLE_TOWER"),          # 'axis' alone is not pitch
    ("Motor current axis 1", "PITCH_HUB"),
    ("Axis1 Motor temperature KTY84", "PITCH_HUB"),
    ("Temperature measured by the IGBT-driver on the rotor side inverter phase1", "ELECTRICAL"),
    ("Drive train vibration axis Z", "ROTOR_DRIVETRAIN"),
    ("Rotor rpm", "ROTOR_DRIVETRAIN"),
    ("Temperature oil in hydraulic group", "HYDRAULIC_BRAKE"),
    ("Transformator core temperature", "ELECTRICAL"),
])
def test_subsystem_rules_farms_a_b(desc, expected):
    assert L.subsystem_of(desc) == expected


def test_care_event_and_score_follow_definition():
    n = 300
    win = np.zeros(n, bool)
    win[100:200] = True
    normal = np.ones(n, bool)
    pred = np.zeros(n, bool)
    pred[100:180] = True                                  # 80 rows flagged inside the window
    a = L.care_event(pred, normal, win, "anomaly")
    assert a["detected"] and a["tp"] == 80 and a["fp"] == 0 and a["fn"] == 20
    assert a["coverage"] == pytest.approx(1.25 * 80 / (1.25 * 80 + 0.25 * 20))
    assert 0.8 < a["earliness"] <= 1.0                    # flagged the early part
    b = L.care_event(np.zeros(n, bool), normal, win, "normal")
    assert b["accuracy"] == 1.0 and not b["detected"]
    s = L.care_score(pd.DataFrame([a, b]))
    assert s["reliability"] == 1.0 and s["detected"] == 1 and s["false_alarms"] == 0
    assert s["care"] == pytest.approx((a["coverage"] + a["earliness"] + 1.0 + 2 * 1.0) / 5)


def test_care_score_special_cases():
    win = np.r_[np.zeros(50, bool), np.ones(50, bool)]
    none = L.care_event(np.zeros(100, bool), np.ones(100, bool), win, "anomaly")
    norm = L.care_event(np.zeros(100, bool), np.ones(100, bool), win, "normal")
    assert L.care_score(pd.DataFrame([none, norm]))["care"] == 0.0     # nothing detected


def test_smooth_is_causal():
    s = np.r_[np.zeros(50), 10.0, np.zeros(49)]
    out = L.smooth(s, rows=5)
    assert (out[:50] == 0).all()                          # no look-ahead
    assert out[50] == 0                                   # one spike does not move a median


def test_condition_normalise_removes_load_dependence():
    rng = np.random.default_rng(0)
    n = 4000
    op = np.column_stack([rng.uniform(0, 20, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n),
                          rng.uniform(0, 1, n)])
    score = np.exp(0.2 * op[:, 0]) * rng.lognormal(0, 0.1, n)      # error grows with wind
    prep = _toy_prepared()
    prep.op, prep.cal_a = op, np.arange(2000)
    out = L.condition_normalise(score, prep)
    assert abs(np.corrcoef(op[2000:, 0], np.log(score[2000:]))[0, 1]) > 0.9
    assert abs(np.corrcoef(op[2000:, 0], out[2000:])[0, 1]) < 0.2


def test_rank_subsystems_rules():
    ratio = np.array([5.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    subs = np.array(["A", "B", "B", "B", "C", "C"])
    assert L.rank_subsystems(ratio, subs, "mean")[0] == "A"
    assert L.rank_subsystems(ratio, subs, "sum")[0] == "A"          # 5 > 4 > 3
    assert L.rank_subsystems(ratio, subs, "vote", top_k=3)[0] == "C"


def test_prepare_drops_counters_and_filters_idle_rows():
    n = 400
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "time_stamp": pd.date_range("2023-01-01", periods=n, freq="10min"),
        "asset_id": 1, "id": np.arange(n),
        "train_test": np.where(np.arange(n) < 300, "train", "prediction"),
        "status_type_id": 0,
        "wind_speed_236_avg": rng.uniform(6, 12, n),
        "power_6_avg": rng.uniform(500, 2000, n),
        "sensor_144_avg": rng.normal(10, 1, n), "sensor_7_avg": rng.normal(10, 1, n),
        "sensor_0_avg": np.cumsum(rng.uniform(0, 1, n)),
    })
    df.loc[10:29, "power_6_avg"] = 0.0                     # idle in good wind
    feats = pd.DataFrame({"description": ["energy counter"], "is_angle": ["False"],
                          "is_counter": ["True"]}, index=["sensor_0"])
    p = L.prepare(df, feats, farm="C", train_filter=True)
    assert "sensor_0_avg" not in p.columns
    assert p.n_filtered == 20 and not set(range(10, 30)) & set(p.fit_idx)
    assert len(p.cal_a) + len(p.cal_b) == len(p.val_idx)


# ----------------------------------------------------------------- v3 additions

def test_care_event_steps_override():
    n = 200
    win = np.r_[np.zeros(100, bool), np.ones(100, bool)]
    pred = np.r_[np.zeros(100, bool), np.ones(50, bool), np.zeros(50, bool)]
    assert not L.care_event(pred, np.ones(n, bool), win, "anomaly")["detected"]   # default 72
    assert not L.care_event(pred, np.ones(n, bool), win, "anomaly", steps=60)["detected"]
    assert L.care_event(pred, np.ones(n, bool), win, "anomaly", steps=36)["detected"]


def test_split_key_handles_seedless_and_underscored_names():
    assert L.split_key("ae_s2") == ("ae", 2)
    assert L.split_key("ae_nofilter_s0") == ("ae_nofilter", 0)
    assert L.split_key("iforest_s1") == ("iforest", 1)
    assert L.split_key("three_sigma") == ("three_sigma", 0)
    assert L.split_key("pca") == ("pca", 0)


def test_corrupt_touches_only_the_given_rows():
    X = np.ones((10, 4), np.float32)
    fill = np.zeros(4, np.float32)
    rows = np.arange(5, 10)
    n = L.corrupt(X, rows, "noise", 0.5, fill)
    assert (n[:5] == 1).all() and not np.allclose(n[5:], 1)
    m = L.corrupt(X, rows, "missing", 1.0, fill)
    assert (m[:5] == 1).all() and (m[5:] == 0).all()
    assert (X == 1).all()                                  # input left unchanged
    with pytest.raises(ValueError):
        L.corrupt(X, rows, "spikes", 0.1, fill)


def test_zscore_ratio_points_at_the_shifted_channel():
    prep = _toy_prepared()
    X = np.zeros((400, 3), np.float32) + 0.5
    X[300:, 1] = 5.0
    prep.X, prep.columns = X, ["a", "b", "c"]
    ev = pd.Series(dict(event_start_id=300, event_end_id=399))
    r = L.zscore_ratios(X, prep, ev)
    assert np.argmax(r) == 1 and r[1] == pytest.approx(10.0, rel=1e-3)


def test_conditional_autoencoder_trains_and_scores():
    rng = np.random.default_rng(0)
    U = rng.normal(size=(600, 2)).astype(np.float32)
    X = np.column_stack([U[:, 0] * 2, U[:, 1] - U[:, 0], rng.normal(size=600)]).astype(np.float32)
    fn, log = L.train_autoencoder(X[:400], X[400:500], epochs=3, U_fit=U[:400], U_val=U[400:500])
    err = fn(X, U)
    assert err.shape == X.shape and np.isfinite(err).all() and len(log) == 3
    fn2, _ = L.train_autoencoder(X[:400], X[400:500], epochs=1, output="sigmoid")
    assert fn2(X).shape == X.shape


def _events(farm, score):
    """One anomaly + one normal dataset whose detection depends on the alarm counter."""
    a = L.care_event(np.r_[np.zeros(50, bool), np.ones(score, bool), np.zeros(100 - score, bool)],
                     np.ones(150, bool), np.r_[np.zeros(50, bool), np.ones(100, bool)], "anomaly")
    b = L.care_event(np.zeros(150, bool), np.ones(150, bool), np.zeros(150, bool), "normal")
    return pd.DataFrame([dict(a, farm=farm), dict(b, farm=farm)])


def test_lofo_select_never_uses_the_held_out_farm():
    rows = []
    for farm in "ABC":
        for q in (0.9, 0.99):
            # setting q=0.9 is good on A and B, setting q=0.99 is good only on C
            good = (q == 0.9) != (farm == "C")
            rows.append(dict(farm=farm, q=q, steps=72, seed=0, events=_events(farm, 90 if good else 10)))
    sel = L.lofo_select(pd.DataFrame(rows))
    chosen = sel.drop_duplicates("heldout").set_index("heldout")["q"]
    assert chosen["C"] == 0.9            # chosen on A+B, even though 0.99 would win on C
    assert chosen["A"] == 0.9 and chosen["B"] == 0.9
    assert set(sel["heldout"]) == {"A", "B", "C"}


# ----------------------------------------------------------------- run B additions

def test_lstm_autoencoder_scores_every_row_causally():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    score, log = L.train_lstm_autoencoder(X, np.arange(200), np.arange(200, 250), epochs=2,
                                          hidden=8)
    assert score.shape == (300,) and np.isfinite(score).all() and len(log) == 2
    X2 = X.copy()
    X2[280:] += 50.0                                      # change only the future
    s2, _ = L.train_lstm_autoencoder(X2, np.arange(200), np.arange(200, 250), epochs=2,
                                     hidden=8)
    assert np.allclose(score[:280], s2[:280], atol=1e-5)  # earlier scores unaffected
    assert s2[285] > score[285]


def test_load_raw_combines_run_a_and_run_b(tmp_path):
    import json
    from merge_results import load_raw
    rows = dict(ids=np.arange(5), eval_normal=np.ones(5, bool), in_window=np.zeros(5, bool))
    for run, key in (("a", "ae_s0|raw"), ("b", "lstm_s0|raw")):
        d = tmp_path / run / "scores"
        d.mkdir(parents=True)
        json.dump(dict(event_id=7, farm="B", label="normal", thresholds={key: 1.0}),
                  open(d / "B_7.json", "w"))
        np.savez(d / "B_7.npz", **rows, **{key: np.zeros(5)})
    raw = load_raw([str(tmp_path / "a"), str(tmp_path / "b")])
    assert len(raw) == 1
    meta, data = raw[0]
    assert set(meta["thresholds"]) == {"ae_s0|raw", "lstm_s0|raw"} and "lstm_s0|raw" in data
