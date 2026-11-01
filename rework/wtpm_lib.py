"""Shared code for the reworked wind-turbine predictive-maintenance study.

Data: CARE-to-Compare (Zenodo record 15846963), wind farms A, B and C (95 datasets).
Every number reported in the paper must come from functions in this file or the
scripts that import it.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ZENODO_URL = "https://zenodo.org/records/15846963/files/CARE_To_Compare.zip?download=1"
FARMS = ("A", "B", "C")
META_COLS = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]
NORMAL_STATUS = (0, 2)          # CARE README: 0 = normal operation, 2 = idling
ALARM_STEPS = 72                # 72 x 10 min = 12 h, CARE / EnergyFaultDetector convention
VAL_FRACTION = 0.15             # last 15 % of normal training rows -> validation
THRESHOLD_Q = 0.99              # fixed before looking at any test result
SMOOTH_ROWS = 36                # causal rolling median, 6 h
FILTER_WIND = (5.0, 20.0)       # CARE paper training filter: wind in operating range ...
FILTER_POWER_FRAC = 0.02        # ... while power is below 2 % of the training 99th percentile
BETA = 0.5                      # CARE score uses F_0.5
EARLINESS_FLAT = 0.25           # earliness weight 1 for the first quarter of the window

# Operating-condition channels used by the condition-normalised score and the filter.
OPERATING = {
    "A": dict(wind="wind_speed_3_avg", power="power_30_avg", rotor="sensor_52_avg",
              ambient="sensor_0_avg"),
    "B": dict(wind="wind_speed_61_avg", power="power_62_avg", rotor="sensor_25_avg",
              ambient="sensor_8_avg"),
    "C": dict(wind="wind_speed_236_avg", power="power_6_avg", rotor="sensor_144_avg",
              ambient="sensor_7_avg"),
}


def farm_prefix(farm: str) -> str:
    return f"CARE_To_Compare/Wind Farm {farm}/"


FARM_PREFIX = farm_prefix("C")   # kept for older scripts

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


def farm_file(name: str, data_dir: str, farm: str = "C") -> str:
    """Local path to a file of one farm; downloads it if missing.

    name: 'event_info.csv', 'feature_description.csv' or 'datasets/<id>.csv'.
    """
    roots = [os.path.join(data_dir, farm_prefix(farm))]
    if farm == "C":
        roots.insert(0, data_dir)
    for root in roots:
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    return fetch_member(farm_prefix(farm) + name, data_dir)


def dataset_path(event_id: int, data_dir: str, farm: str = "C") -> str:
    return os.path.join(data_dir, farm_prefix(farm), "datasets", f"{event_id}.csv")


def load_event_info(data_dir: str, farm: str = "C") -> pd.DataFrame:
    ev = pd.read_csv(farm_file("event_info.csv", data_dir, farm), sep=";")
    ev["event_start"] = pd.to_datetime(ev["event_start"])
    ev["event_end"] = pd.to_datetime(ev["event_end"])
    ev["farm"] = farm
    return ev.sort_values("event_id").reset_index(drop=True)


def load_features(data_dir: str, farm: str = "C") -> pd.DataFrame:
    fd = pd.read_csv(farm_file("feature_description.csv", data_dir, farm), sep=";",
                     encoding="latin-1")
    return fd.drop_duplicates("sensor_name").set_index("sensor_name")


def load_dataset(event_id: int, data_dir: str, farm: str = "C") -> pd.DataFrame:
    df = pd.read_csv(farm_file(f"datasets/{event_id}.csv", data_dir, farm), sep=";")
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    return df.sort_values("id").reset_index(drop=True)


def base_sensor(col: str) -> str:
    """'sensor_12_avg' -> 'sensor_12'; 'wind_speed_236_std' -> 'wind_speed_236'."""
    return re.sub(r"_(avg|min|max|std)$", "", col)


def _flag(features: pd.DataFrame, column: str) -> set[str]:
    if column not in features.columns:
        return set()
    return set(features.index[features[column].astype(str).str.lower() == "true"])

# ------------------------------------------------------- subsystem mapping
# Built from the human-readable descriptions in feature_description.csv of each farm.
# Regex rules are applied in order; the first match wins.
SUBSYSTEM_RULES: list[tuple[str, str]] = [
    ("PITCH_HUB", r"axis ?\d|blade|pitch|\bhub\b|spinner|battery|nose cone|split ring"),
    ("YAW", r"yaw|azimuth"),
    ("HYDRAULIC_BRAKE", r"hydraulic|rotor brake|locking|oil container|filter pump"),
    ("COOLING", r"cooling|cooler|water|aeration|aerator"),
    ("GEARBOX", r"gear|planetary"),
    ("GENERATOR", r"generator|stator winding"),
    ("ROTOR_DRIVETRAIN", r"rotor bearing|rotor speed|rotor rpm|axial bearing|main shaft|"
                         r"drive ?train|torque"),
    ("NACELLE_TOWER", r"nacelle vibration|tower|platform|nacelle temperature|top nacelle"),
    ("ELECTRICAL", r"converter|inverter|igbt|abb-ls|grid|mains|internal consumption|"
                   r"transformer|transformator|24 ?v|power pack|cabinet|control box|board|"
                   r"current|reactive power|active power|apparent power|available power|"
                   r"compensation angle|dc link|voltage|\brms\b|choke|busbar|"
                   r"phase displacement|cable load|vcp"),
    ("ENVIRONMENT", r"ambient|outside|wind|air pressure"),
]


def subsystem_of(description: str) -> str:
    d = description.lower()
    for name, pattern in SUBSYSTEM_RULES:
        if re.search(pattern, d):
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
# Event ids are unique across the three farms. None = description too vague
# (communication faults, unknown standstill, mixed safety items).
EVENT_SUBSYSTEMS: dict[int, list[str] | None] = {
    # farm C
    55: ["PITCH_HUB"], 81: ["ELECTRICAL"], 47: ["HYDRAULIC_BRAKE"],
    12: ["GEARBOX", "HYDRAULIC_BRAKE", "YAW"], 4: ["PITCH_HUB"], 18: None,
    28: ["HYDRAULIC_BRAKE", "PITCH_HUB"], 39: ["GEARBOX"], 66: ["PITCH_HUB"],
    15: ["PITCH_HUB"], 78: None, 79: None, 30: ["PITCH_HUB"], 33: ["PITCH_HUB"],
    11: None, 44: ["COOLING"], 49: ["GEARBOX"], 31: ["PITCH_HUB"], 67: ["ELECTRICAL"],
    9: ["YAW"], 91: ["PITCH_HUB"], 5: ["ELECTRICAL"], 90: None, 70: ["PITCH_HUB"],
    35: ["ELECTRICAL"], 16: ["PITCH_HUB"], 76: ["PITCH_HUB"],
    # farm A (EDP logbook)
    68: ["ELECTRICAL"], 22: ["HYDRAULIC_BRAKE"], 72: ["GEARBOX"], 73: ["HYDRAULIC_BRAKE"],
    0: ["GENERATOR"], 26: ["HYDRAULIC_BRAKE"], 40: ["GENERATOR"], 42: ["HYDRAULIC_BRAKE"],
    10: ["GEARBOX"], 45: ["HYDRAULIC_BRAKE"], 84: ["HYDRAULIC_BRAKE"], 51: ["GEARBOX"],
    # farm B
    34: ["ELECTRICAL"], 7: ["ELECTRICAL"], 19: ["ELECTRICAL"], 53: ["ROTOR_DRIVETRAIN"],
    27: ["ROTOR_DRIVETRAIN"], 77: ["ROTOR_DRIVETRAIN"],
}

# ---------------------------------------------------------- preprocessing

@dataclass
class Prepared:
    X: np.ndarray            # standardised features, all rows, float32
    columns: list[str]
    fit_idx: np.ndarray      # rows used to fit models
    val_idx: np.ndarray      # rows used only for early stopping / calibration / thresholds
    pred_mask: np.ndarray    # prediction-window rows
    normal_mask: np.ndarray  # status 0 or 2 (used to choose training rows)
    df: pd.DataFrame         # raw frame (unscaled), same row order
    eval_normal: np.ndarray | None = None   # rows that count in evaluation
    op: np.ndarray | None = None            # raw operating conditions (rows x 4)
    n_filtered: int = 0
    cal_a: np.ndarray = field(default_factory=lambda: np.array([], int))
    cal_b: np.ndarray = field(default_factory=lambda: np.array([], int))


def operating_matrix(df: pd.DataFrame, farm: str, rows: np.ndarray) -> np.ndarray:
    cols = OPERATING[farm]
    Z = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(float)
                         if c in df.columns else np.full(len(df), np.nan)
                         for c in cols.values()])
    med = np.nanmedian(Z[rows], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(Z), med, Z)


def prepare(df: pd.DataFrame, features: pd.DataFrame, max_missing: float = 0.30,
            farm: str = "C", train_filter: bool = False) -> Prepared:
    """Leakage rules: every statistic is fitted on normal-status TRAINING rows only.

    Dropped: metadata columns (id, asset_id, status_type_id ...), angle sensors (wrap at 360),
    counter sensors (cumulative energy), columns with > max_missing NaN in training, and
    columns constant in training. With train_filter, training rows where the wind is in the
    operating range but the turbine produces (almost) no power are also left out.
    Farm A's status codes come from the fault logbook, so they are used to pick training rows
    but are ignored in evaluation (as in the CARE reference implementation).
    """
    normal = df["status_type_id"].isin(NORMAL_STATUS).to_numpy()
    train = (df["train_test"] == "train").to_numpy()
    candidate = train & normal
    n_filtered = 0
    if train_filter and farm in OPERATING:
        op_cols = OPERATING[farm]
        wind = pd.to_numeric(df.get(op_cols["wind"]), errors="coerce").to_numpy(float)
        power = pd.to_numeric(df.get(op_cols["power"]), errors="coerce").to_numpy(float)
        p99 = np.nanpercentile(power[candidate], 99)
        idle = (wind >= FILTER_WIND[0]) & (wind <= FILTER_WIND[1]) & \
               (power < FILTER_POWER_FRAC * p99)
        n_filtered = int((candidate & idle).sum())
        candidate &= ~idle
    tr_idx = np.flatnonzero(candidate)
    n_val = int(len(tr_idx) * VAL_FRACTION)
    fit_idx, val_idx = tr_idx[:-n_val], tr_idx[-n_val:]
    half = len(val_idx) // 2

    cols = [c for c in df.columns if c not in META_COLS]
    skip = _flag(features, "is_angle") | _flag(features, "is_counter")
    cols = [c for c in cols if base_sensor(c) not in skip]
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
    eval_normal = np.ones(len(df), bool) if farm == "A" else normal
    op = operating_matrix(df, farm, fit_idx) if farm in OPERATING else None
    return Prepared(X, cols, fit_idx, val_idx, ~train, normal, df, eval_normal, op,
                    n_filtered, val_idx[:half], val_idx[half:])

# ------------------------------------------------------------------ models

def _build_ae(d: int, c: int = 0, output: str = "linear"):
    """Dense AE d-256-64-16-64-256-d. With c > 0 it is a conditional AE: the c operating
    conditions are appended to the encoder input and to the code before decoding, so the
    network only has to explain what the operating point does not."""
    from torch import nn

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d + c, 256), nn.ReLU(), nn.Linear(256, 64),
                                     nn.ReLU(), nn.Linear(64, 16), nn.ReLU())
            dec = [nn.Linear(16 + c, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(),
                   nn.Linear(256, d)]
            if output == "sigmoid":
                dec.append(nn.Sigmoid())
            self.dec = nn.Sequential(*dec)

        def forward(self, x, u=None):
            import torch
            if c:
                return self.dec(torch.cat([self.enc(torch.cat([x, u], 1)), u], 1))
            return self.dec(self.enc(x))

    return AE()


def train_autoencoder(X_fit, X_val, seed=0, output="linear", epochs=40, batch=256,
                      lr=1e-3, patience=5, device=None, U_fit=None, U_val=None):
    """Train the AE (conditional if U_fit is given). Returns a channel-error function
    errfn(X, U=None) and the training log [(epoch, train_loss, val_loss)]."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = X_fit.shape[1]
    c = 0 if U_fit is None else U_fit.shape[1]
    model = _build_ae(d, c, output).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xf = torch.from_numpy(X_fit).to(device)
    xv = torch.from_numpy(X_val).to(device)
    uf = torch.from_numpy(U_fit.astype(np.float32)).to(device) if c else None
    uv = torch.from_numpy(U_val.astype(np.float32)).to(device) if c else None
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, wait, log = math.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xf), generator=gen)
        tot = 0.0
        for i in range(0, len(xf), batch):
            j = perm[i:i + batch].to(device)
            xb = xf[j]
            loss = ((model(xb, uf[j] if c else None) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vl = ((model(xv, uv) - xv) ** 2).mean().item()
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

    def channel_errors(X, U=None):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                xb = torch.from_numpy(X[i:i + 8192]).to(device)
                ub = torch.from_numpy(U[i:i + 8192].astype(np.float32)).to(device) if c else None
                out.append(((model(xb, ub) - xb) ** 2).cpu().numpy())
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

# ------------------------------------------------------- score post-processing

def smooth(score: np.ndarray, rows: int = SMOOTH_ROWS) -> np.ndarray:
    """Causal rolling median (uses only the current and past rows)."""
    return pd.Series(score).rolling(rows, min_periods=1).median().to_numpy()


def condition_normalise(score: np.ndarray, prep: Prepared, seed: int = 0) -> np.ndarray:
    """Remove the part of the score that operating conditions explain.

    A gradient-boosting model learns log(score) from wind speed, power, rotor speed and
    ambient temperature on the first half of the validation rows (rows the detector never
    trained on). The normalised score is log(score) minus that expectation.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = np.log(np.maximum(score, 0) + 1e-6)
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=50,
                                      random_state=seed)
    m.fit(prep.op[prep.cal_a], y[prep.cal_a])
    return y - m.predict(prep.op)


VARIANTS = ("raw", "smooth", "cn", "cn_smooth")


def variants(score: np.ndarray, prep: Prepared) -> dict[str, np.ndarray]:
    cn = condition_normalise(score, prep)
    return {"raw": score, "smooth": smooth(score), "cn": cn, "cn_smooth": smooth(cn)}

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


def fbeta(tp: float, fp: float, fn: float, beta: float = BETA) -> float:
    b2 = beta * beta
    den = (1 + b2) * tp + b2 * fn + fp
    return float((1 + b2) * tp / den) if den > 0 and tp > 0 else 0.0


def earliness_weights(n: int) -> np.ndarray:
    if n == 1:
        return np.ones(1)
    pos = np.arange(n) / (n - 1)
    return np.where(pos < EARLINESS_FLAT, 1.0, (1.0 - pos) / (1.0 - EARLINESS_FLAT))


def care_event(pred: np.ndarray, normal: np.ndarray, in_window: np.ndarray, label: str,
               steps: int | None = None) -> dict:
    """Score one dataset's prediction window with the CARE definitions (Gueck et al. 2024),
    following the reference implementation (EnergyFaultDetector) where the paper is open:
    the criticality is counted up to the end of the event window, earliness weight is 1 for
    the first quarter of the window and falls linearly to 0, abnormal-status rows are left out
    of coverage and accuracy."""
    steps = ALARM_STEPS if steps is None else steps
    pred, normal, in_window = (np.asarray(a, bool) for a in (pred, normal, in_window))
    win = np.flatnonzero(in_window)
    stop = win[-1] + 1 if len(win) else len(pred)
    crit = criticality(pred[:stop], normal[:stop])
    truth = in_window if label == "anomaly" else np.zeros(len(pred), bool)
    p, t = pred[normal], truth[normal]
    tp, fp = int((p & t).sum()), int((p & ~t).sum())
    fn, tn = int((~p & t).sum()), int((~p & ~t).sum())
    out = dict(label=label, tp=tp, fp=fp, fn=fn, tn=tn,
               accuracy=(tp + tn) / max(tp + fp + fn + tn, 1),
               coverage=fbeta(tp, fp, fn) if label == "anomaly" else np.nan,
               max_crit=int(crit.max()) if len(crit) else 0,
               detected=bool(len(crit) and crit.max() >= steps),
               earliness=np.nan, lead_h=np.nan)
    if label == "anomaly" and len(win):
        w = earliness_weights(len(win))
        out["earliness"] = float((w * pred[win]).sum() / w.sum())
        first = np.flatnonzero(crit >= steps)
        out["lead_h"] = (stop - 1 - first[0]) / 6.0 if len(first) else np.nan
    return out


def care_score(events: pd.DataFrame) -> dict:
    """Combine per-dataset rows of care_event() into the CARE score and its parts."""
    an = events["label"] == "anomaly"
    det = events["detected"].astype(bool)
    tp, fp, fn = int((det & an).sum()), int((det & ~an).sum()), int((~det & an).sum())
    parts = dict(coverage=float(events.loc[an, "coverage"].mean()),
                 accuracy=float(events.loc[~an, "accuracy"].mean()),
                 reliability=fbeta(tp, fp, fn),
                 earliness=float(events.loc[an, "earliness"].mean()),
                 detected=tp, n_anomaly=int(an.sum()), false_alarms=fp, n_normal=int((~an).sum()))
    if tp + fp == 0:
        care = 0.0
    elif parts["accuracy"] <= 0.5:
        care = parts["accuracy"]
    else:
        care = (parts["coverage"] + parts["earliness"] + parts["reliability"]
                + 2 * parts["accuracy"]) / 5
    return dict(care=float(care), **parts)


def evaluate(score, prep: Prepared, ev: pd.Series, threshold: float) -> dict:
    """Event-level and point-wise metrics inside the prediction window of one dataset (v1)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    df = prep.df
    pm = prep.pred_mask
    s = score[pm]
    nm = (prep.eval_normal if prep.eval_normal is not None else prep.normal_mask)[pm]
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


def channel_ratios(ch_err: np.ndarray, prep: Prepared, ev: pd.Series) -> np.ndarray | None:
    """Each channel's mean error over the event window divided by its mean validation error."""
    base = ch_err[prep.val_idx].mean(axis=0) + 1e-8
    ids = prep.df["id"].to_numpy()
    nm = prep.eval_normal if prep.eval_normal is not None else prep.normal_mask
    win = (ids >= ev["event_start_id"]) & (ids <= ev["event_end_id"]) & nm
    if win.sum() == 0:
        return None
    return ch_err[win].mean(axis=0) / base


def rank_subsystems(ratio: np.ndarray, subs: np.ndarray, rule: str, top_k: int = 10) -> list[str]:
    """Rank plant areas by channel error ratios.

    mean: mean ratio per area (size-neutral); sum: summed ratio (favours big areas);
    vote: areas ranked by how many of the top_k channels they own (ties by mean ratio)."""
    names = np.unique(subs)
    mean = {n: ratio[subs == n].mean() for n in names}
    if rule == "mean":
        key = mean
    elif rule == "sum":
        key = {n: ratio[subs == n].sum() for n in names}
    elif rule == "vote":
        top = subs[np.argsort(ratio)[::-1][:top_k]]
        key = {n: (int((top == n).sum()), mean[n]) for n in names}
    else:
        raise ValueError(rule)
    return sorted(names, key=lambda n: key[n], reverse=True)


def subsystem_scores(ch_err: np.ndarray, prep: Prepared, ev: pd.Series, col_sub: dict) -> pd.DataFrame:
    """Per-subsystem attribution over the event window (mean and summed ratios)."""
    ratio = channel_ratios(ch_err, prep, ev)
    if ratio is None:
        return pd.DataFrame()
    sub = np.array([col_sub[c] for c in prep.columns])
    rows = []
    for name in np.unique(sub):
        r = ratio[sub == name]
        rows.append(dict(subsystem=name, mean_ratio=float(r.mean()), sum_ratio=float(r.sum()),
                         n_channels=int(len(r))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ v3 additions

def scaled_conditions(prep: Prepared) -> np.ndarray:
    """Operating conditions standardised on the fit rows (input of the conditional AE)."""
    mu = prep.op[prep.fit_idx].mean(axis=0)
    sd = prep.op[prep.fit_idx].std(axis=0) + 1e-8
    return np.clip((prep.op - mu) / sd, -10, 10).astype(np.float32)


CORRUPTIONS = (("noise", 0.1), ("noise", 0.3), ("noise", 0.5), ("missing", 0.05),
               ("missing", 0.2))


def corrupt(X: np.ndarray, rows: np.ndarray, kind: str, level: float, fill: np.ndarray,
            seed: int = 0) -> np.ndarray:
    """Copy of X with the given rows corrupted: Gaussian noise of sd `level` (standardised
    units), or a fraction `level` of entries replaced by `fill` (the training median), which
    is what the preprocessing does with a missing value."""
    rng = np.random.default_rng(seed)
    Xc = X.copy()
    block = Xc[rows]
    if kind == "noise":
        block = block + rng.normal(0, level, block.shape).astype(np.float32)
    elif kind == "missing":
        mask = rng.random(block.shape) < level
        block = np.where(mask, np.broadcast_to(fill, block.shape), block).astype(np.float32)
    else:
        raise ValueError(kind)
    Xc[rows] = block
    return Xc


def zscore_ratios(X: np.ndarray, prep: Prepared, ev: pd.Series) -> np.ndarray | None:
    """Attribution baseline without any model: each channel's mean |z| over the event window
    divided by its mean |z| on the validation rows."""
    ids = prep.df["id"].to_numpy()
    nm = prep.eval_normal if prep.eval_normal is not None else prep.normal_mask
    win = (ids >= ev["event_start_id"]) & (ids <= ev["event_end_id"]) & nm
    if win.sum() == 0:
        return None
    return np.abs(X[win]).mean(axis=0) / (np.abs(X[prep.val_idx]).mean(axis=0) + 1e-8)


def split_key(base: str) -> tuple[str, int]:
    """'ae_s2' -> ('ae', 2); 'iforest_s1' -> ('iforest', 1); 'pca' -> ('pca', 0)."""
    m = re.match(r"^(.*)_s(\d+)$", base)
    return (m.group(1), int(m.group(2))) if m else (base, 0)


TUNE_Q = (0.95, 0.975, 0.99, 0.995, 0.999)
TUNE_STEPS = (36, 72, 144)


def lofo_select(table: pd.DataFrame, farms=FARMS) -> pd.DataFrame:
    """Leave-one-farm-out choice of (q, steps).

    table: one row per (farm, q, steps, seed) with the per-event rows of that setting in
    column 'events' (a DataFrame). For each held-out farm, picks the setting with the best
    seed-averaged CARE score on the other farms and returns the held-out farm's per-event rows
    under that setting (all seeds), with the chosen setting attached."""
    out = []
    for f in farms:
        if f not in set(table["farm"]):
            continue
        train = table[table["farm"] != f]
        best, best_care = None, -1.0
        for (q, st), g in train.groupby(["q", "steps"]):
            cares = [care_score(pd.concat(gs["events"].tolist()))["care"]
                     for _, gs in g.groupby("seed")]
            c = float(np.nanmean(cares)) if np.isfinite(cares).any() else -np.inf
            if c > best_care + 1e-12:
                best, best_care = (q, st), c
        if best is None:
            continue
        test = table[(table["farm"] == f) & (table["q"] == best[0]) & (table["steps"] == best[1])]
        for _, r in test.iterrows():
            out.append(r["events"].assign(heldout=f, q=best[0], steps=best[1], seed=r["seed"],
                                          train_care=best_care))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------- run B: LSTM autoencoder

LSTM_WINDOW = 6        # 1 h of 10-min rows, causal (the window ends at the scored row)


def train_lstm_autoencoder(X, fit_idx, val_idx, seed=0, window=LSTM_WINDOW, hidden=64,
                           epochs=40, batch=256, lr=1e-3, patience=5, device=None):
    """Sequence AE: an LSTM encodes the last `window` rows, its final state is repeated and a
    second LSTM decodes it back to the window. Row t is scored by the mean squared error of
    the window that ends at t (rows before the start repeat the first row), so the score
    never looks ahead. Trained on windows that end on fit rows; early stopping on windows
    that end on validation rows. Returns (score over all rows, training log)."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n, d = X.shape

    class SeqAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.LSTM(d, hidden, batch_first=True)
            self.dec = nn.LSTM(hidden, hidden, batch_first=True)
            self.out = nn.Linear(hidden, d)

        def forward(self, x):
            _, (h, _) = self.enc(x)
            z = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
            y, _ = self.dec(z)
            return self.out(y)

    model = SeqAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(X).to(device)
    offs = torch.arange(-window + 1, 1, device=device)

    def windows(ends):
        idx = (ends.unsqueeze(1) + offs).clamp(min=0)
        return xt[idx]

    fit_e = torch.from_numpy(np.asarray(fit_idx)).to(device)
    val_e = torch.from_numpy(np.asarray(val_idx)).to(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def mse_of(ends):
        out = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(ends), 4096):
                w = windows(ends[i:i + 4096])
                out.append(((model(w) - w) ** 2).mean(dim=(1, 2)).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    best, best_state, wait, log = math.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = fit_e[torch.randperm(len(fit_e), generator=gen).to(device)]
        tot = 0.0
        for i in range(0, len(perm), batch):
            w = windows(perm[i:i + batch])
            loss = ((model(w) - w) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(w)
        vl = float(mse_of(val_e).mean())
        log.append((ep, tot / len(fit_e), vl))
        if vl < best - 1e-6:
            best, wait = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return mse_of(torch.arange(n, device=device)), log
