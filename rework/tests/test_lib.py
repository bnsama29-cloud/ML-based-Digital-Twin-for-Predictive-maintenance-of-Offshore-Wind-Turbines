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
