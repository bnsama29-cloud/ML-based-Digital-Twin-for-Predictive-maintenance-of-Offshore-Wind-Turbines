"""Shared code for the reworked wind-turbine predictive-maintenance study.

Data: CARE-to-Compare (Zenodo record 15846963), Wind Farm C (957 columns, 10-min SCADA).
Every number reported in the paper must come from functions in this file or the
scripts that import it.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

ZENODO_URL = "https://zenodo.org/records/15846963/files/CARE_To_Compare.zip?download=1"
FARM_PREFIX = "CARE_To_Compare/Wind Farm C/"
META_COLS = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]
NORMAL_STATUS = (0, 2)          # CARE README: 0 = normal operation, 2 = idling
ALARM_STEPS = 72                # 72 x 10 min = 12 h, CARE / EnergyFaultDetector convention
VAL_FRACTION = 0.15             # last 15 % of normal training rows -> validation
THRESHOLD_Q = 0.99              # fixed before looking at any test result

# ---------------------------------------------------------------- data access

def fetch_member(member: str, dest_dir: str, attempts: int = 5, wait_s: float = 30.0) -> str:
    """Extract one file from the Zenodo zip with HTTP range requests (no 5.5 GB download).

    Downloads into a temporary folder and moves the file into place only when complete,
    so a dropped connection never leaves a half-written CSV behind. Retries with backoff.
    """
    import shutil
    import tempfile
    import time

    from remotezip import RemoteZip
    out = os.path.join(dest_dir, member)
    if os.path.exists(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(1, attempts + 1):
        tmp = tempfile.mkdtemp(dir=dest_dir)
        try:
            with RemoteZip(ZENODO_URL, timeout=(20, 300)) as z:
                z.extract(member, tmp)
            shutil.move(os.path.join(tmp, member), out)
            return out
        except Exception as err:  # network errors, HTTP 429/5xx, truncated reads
            if attempt == attempts:
                raise
            print(f"download of {member} failed ({err!r}); retry {attempt}/{attempts - 1}",
                  flush=True)
            time.sleep(wait_s * attempt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return out


def farm_file(name: str, data_dir: str) -> str:
    """Local path to a farm C file; downloads it if missing.

    name: 'event_info.csv', 'feature_description.csv' or 'datasets/<id>.csv'.
    """
    for root in (data_dir, os.path.join(data_dir, FARM_PREFIX)):
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    return fetch_member(FARM_PREFIX + name, data_dir)


def load_event_info(data_dir: str) -> pd.DataFrame:
    ev = pd.read_csv(farm_file("event_info.csv", data_dir), sep=";")
    ev["event_start"] = pd.to_datetime(ev["event_start"])
    ev["event_end"] = pd.to_datetime(ev["event_end"])
    return ev.sort_values("event_id").reset_index(drop=True)


def load_features(data_dir: str) -> pd.DataFrame:
    fd = pd.read_csv(farm_file("feature_description.csv", data_dir), sep=";")
    return fd.drop_duplicates("sensor_name").set_index("sensor_name")


def load_dataset(event_id: int, data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(farm_file(f"datasets/{event_id}.csv", data_dir), sep=";")
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    return df.sort_values("id").reset_index(drop=True)


def base_sensor(col: str) -> str:
    """'sensor_12_avg' -> 'sensor_12'; 'wind_speed_236_std' -> 'wind_speed_236'."""
    return re.sub(r"_(avg|min|max|std)$", "", col)

# ------------------------------------------------------- subsystem mapping
# Built from the human-readable descriptions in feature_description.csv.
# Rules are applied in order; the first match wins.
SUBSYSTEM_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("PITCH_HUB", ("axis", "blade", "pitch", "hub", "spinner", "battery")),
    ("YAW", ("yaw", "azimuth")),
    ("HYDRAULIC_BRAKE", ("hydraulic", "rotor brake", "locking", "oil container", "filter pump")),
    ("COOLING", ("cooling", "cooler", "water", "aeration", "aerator")),
    ("GEARBOX", ("gear", "planetary")),
    ("GENERATOR", ("generator", "stator winding")),
    ("ROTOR_DRIVETRAIN", ("rotor bearing", "rotor speed", "axial bearing", "main shaft")),
    ("ELECTRICAL", ("converter", "abb-ls", "grid", "mains", "internal consumption", "transformer",
                    "24v", "24 vdc", "24vdc", "power pack", "cabinet", "control box", "board",
                    "current drain", "reactive power", "active power", "compensation angle",
                    "dc link", "voltage", "rms")),
    ("NACELLE_TOWER", ("nacelle vibration", "platform")),
    ("ENVIRONMENT", ("ambient", "outside", "wind", "air pressure")),
]


def subsystem_of(description: str) -> str:
    d = description.lower()
    for name, keys in SUBSYSTEM_RULES:
        if any(k in d for k in keys):
            return name
    return "OTHER"


def column_subsystems(columns: list[str], features: pd.DataFrame) -> dict[str, str]:
    out = {}
    for c in columns:
        s = base_sensor(c)
        desc = features["description"].get(s, "") if s in features.index else ""
        out[c] = subsystem_of(str(desc))
    return out


# Ground truth for attribution, written by hand from event_info.csv 'event_description'.
# None = description too vague (communication faults, unknown standstill, mixed safety items).
EVENT_SUBSYSTEMS: dict[int, list[str] | None] = {
    55: ["PITCH_HUB"], 81: ["ELECTRICAL"], 47: ["HYDRAULIC_BRAKE"],
    12: ["GEARBOX", "HYDRAULIC_BRAKE", "YAW"], 4: ["PITCH_HUB"], 18: None,
    28: ["HYDRAULIC_BRAKE", "PITCH_HUB"], 39: ["GEARBOX"], 66: ["PITCH_HUB"],
    15: ["PITCH_HUB"], 78: None, 79: None, 30: ["PITCH_HUB"], 33: ["PITCH_HUB"],
    11: None, 44: ["COOLING"], 49: ["GEARBOX"], 31: ["PITCH_HUB"], 67: ["ELECTRICAL"],
    9: ["YAW"], 91: ["PITCH_HUB"], 5: ["ELECTRICAL"], 90: None, 70: ["PITCH_HUB"],
    35: ["ELECTRICAL"], 16: ["PITCH_HUB"], 76: ["PITCH_HUB"],
}

# ---------------------------------------------------------- preprocessing

@dataclass
class Prepared:
    X: np.ndarray            # standardised features, all rows, float32
    columns: list[str]
    fit_idx: np.ndarray      # rows used to fit models
    val_idx: np.ndarray      # rows used only to set thresholds / early stopping
    pred_mask: np.ndarray    # prediction-window rows
    normal_mask: np.ndarray  # status 0 or 2
    df: pd.DataFrame         # raw frame (unscaled), same row order


def prepare(df: pd.DataFrame, features: pd.DataFrame, max_missing: float = 0.30) -> Prepared:
    """Leakage rules: every statistic is fitted on normal-status TRAINING rows only.

    Dropped: metadata columns (id, asset_id, status_type_id ...), angle sensors (wrap at 360),
    columns with > max_missing NaN in training, and columns constant in training.
    """
    normal = df["status_type_id"].isin(NORMAL_STATUS).to_numpy()
    train = (df["train_test"] == "train").to_numpy()
    tr_idx = np.flatnonzero(train & normal)
    n_val = int(len(tr_idx) * VAL_FRACTION)
    fit_idx, val_idx = tr_idx[:-n_val], tr_idx[-n_val:]

    cols = [c for c in df.columns if c not in META_COLS]
    angle = set(features.index[features["is_angle"].astype(str).str.lower() == "true"])
    cols = [c for c in cols if base_sensor(c) not in angle]
    raw = df[cols].apply(pd.to_numeric, errors="coerce")
    miss = raw.iloc[fit_idx].isna().mean()
    cols = [c for c in cols if miss[c] <= max_missing]
    raw = raw[cols].ffill(limit=6)                      # causal gap fill, max 1 h
    med = raw.iloc[fit_idx].median()
    raw = raw.fillna(med)
    mu = raw.iloc[fit_idx].mean()
    sd = raw.iloc[fit_idx].std()
    keep = sd > 1e-8
    cols = [c for c in cols if keep[c]]
    X = ((raw[cols] - mu[cols]) / sd[cols]).clip(-10, 10).to_numpy(np.float32)
    return Prepared(X, cols, fit_idx, val_idx, ~train, normal, df)

# ------------------------------------------------------------------ models

def train_autoencoder(X_fit, X_val, seed=0, output="linear", epochs=40, batch=256,
                      lr=1e-3, patience=5, device=None):
    """Dense AE d-256-64-16-64-256-d. Returns a scoring function and training log."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = X_fit.shape[1]
    layers = [nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 16),
              nn.ReLU(), nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(),
              nn.Linear(256, d)]
    if output == "sigmoid":
        layers.append(nn.Sigmoid())
    model = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xf = torch.from_numpy(X_fit).to(device)
    xv = torch.from_numpy(X_val).to(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, wait, log = math.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xf), generator=gen)
        tot = 0.0
        for i in range(0, len(xf), batch):
            xb = xf[perm[i:i + batch].to(device)]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vl = ((model(xv) - xv) ** 2).mean().item()
        log.append((ep, tot / len(xf), vl))
        if vl < best - 1e-6:
            best, wait = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    def channel_errors(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                xb = torch.from_numpy(X[i:i + 8192]).to(device)
                out.append(((model(xb) - xb) ** 2).cpu().numpy())
        return np.concatenate(out)

    return channel_errors, log


def score_isolation_forest(X_fit, X, seed=0):
    from sklearn.ensemble import IsolationForest
    m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(X_fit)
    return -m.score_samples(X)


def score_pca(X_fit, X, var=0.95):
    from sklearn.decomposition import PCA
    m = PCA(n_components=var, svd_solver="full").fit(X_fit)
    return ((m.inverse_transform(m.transform(X)) - X) ** 2).mean(axis=1)


def score_three_sigma(X):
    """Fraction of channels beyond 3 sd of the training distribution (X is already z-scored)."""
    return (np.abs(X) > 3).mean(axis=1)

# ------------------------------------------------------------- evaluation

def criticality(anomaly: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """CARE/EnergyFaultDetector counter: +1 on anomalous normal-status rows, -1 (floored at 0)
    on clean normal-status rows, unchanged on abnormal-status rows."""
    c = np.zeros(len(anomaly), dtype=np.int32)
    v = 0
    for i, (a, n) in enumerate(zip(anomaly, normal)):
        if n:
            v = v + 1 if a else max(v - 1, 0)
        c[i] = v
    return c


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def evaluate(score, prep: Prepared, ev: pd.Series, threshold: float) -> dict:
    """Event-level and point-wise metrics inside the prediction window of one dataset."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    df = prep.df
    pm = prep.pred_mask
    s = score[pm]
    nm = prep.normal_mask[pm]
    ids = df["id"].to_numpy()[pm]
    ts = df["time_stamp"].to_numpy()[pm]
    anom = s > threshold
    crit = criticality(anom, nm)
    alarm = crit >= ALARM_STEPS
    is_anom_event = ev["event_label"] == "anomaly"
    in_event = (ids >= ev["event_start_id"]) & (ids <= ev["event_end_id"])
    y = in_event if is_anom_event else np.zeros_like(in_event)
    yy, pp, ss = y[nm], anom[nm], s[nm]
    rec = dict(
        event_id=int(ev["event_id"]), label=ev["event_label"],
        tp=int((yy & pp).sum()), fp=int((~yy & pp).sum()),
        fn=int((yy & ~pp).sum()), tn=int((~yy & ~pp).sum()),
        alarm_any=bool(alarm.any()),
        detected=bool((alarm & in_event).any()) if is_anom_event else np.nan,
        roc_auc=np.nan, pr_auc=np.nan, first_alarm_h_rel_start=np.nan,
    )
    if is_anom_event and 0 < yy.sum() < len(yy):
        rec["roc_auc"] = float(roc_auc_score(yy, ss))
        rec["pr_auc"] = float(average_precision_score(yy, ss))
    if alarm.any():
        t0 = pd.Timestamp(ts[np.argmax(alarm)])
        rec["first_alarm_h_rel_start"] = (t0 - ev["event_start"]).total_seconds() / 3600
    return rec


def subsystem_scores(ch_err: np.ndarray, prep: Prepared, ev: pd.Series, col_sub: dict) -> pd.DataFrame:
    """Per-subsystem attribution over the event window: channel error divided by the
    channel's mean validation error, aggregated by mean (size-neutral) and by sum (paper eq.)."""
    base = ch_err[prep.val_idx].mean(axis=0) + 1e-8
    ids = prep.df["id"].to_numpy()
    win = (ids >= ev["event_start_id"]) & (ids <= ev["event_end_id"]) & prep.normal_mask
    if win.sum() == 0:
        return pd.DataFrame()
    ratio = ch_err[win].mean(axis=0) / base
    sub = np.array([col_sub[c] for c in prep.columns])
    rows = []
    for name in np.unique(sub):
        r = ratio[sub == name]
        rows.append(dict(subsystem=name, mean_ratio=float(r.mean()), sum_ratio=float(r.sum()),
                         n_channels=int(len(r))))
    return pd.DataFrame(rows)
