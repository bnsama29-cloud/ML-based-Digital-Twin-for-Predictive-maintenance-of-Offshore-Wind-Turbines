import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import wtpm_lib as L  # noqa: E402
from analyze_44 import time_to_threshold  # noqa: E402


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
