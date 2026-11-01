"""Everything in one go, on one Kaggle account (v3, the final run):
benchmark on all 95 datasets (farms A, B, C) -> two case studies -> result tables -> zip.

    python run_all.py --data-dir /tmp/care --out /tmp/results --zip /kaggle/working/results_v3.zip

The work happens in --out; at the end (also after a crash or when the time budget runs out)
everything except the large score files is zipped to --zip, and the score files to a second
zip next to it, so the Kaggle Output tab holds just two files.
"""
import argparse
import os
import subprocess
import sys
import time
import zipfile

CASES = [("C", 44, 61),   # cooling-water valve fault (missed by the v2 pipeline, kept as is)
         ("A", 40, 0)]    # generator bearing failure; fixed before the v3 run as the
                          # detected illustration (detected in all three v2 seeds)


def step(name, cmd, failures):
    print(f"\n===== {name} =====", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, *cmd])
    status = "done" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
    if r.returncode:
        failures.append(name)
    print(f"===== {name} {status} in {(time.time() - t0) / 60:.1f} min =====", flush=True)


def make_zips(out, zip_path):
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
    scores_zip = zip_path.replace(".zip", "_scores.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zr, \
            zipfile.ZipFile(scores_zip, "w", zipfile.ZIP_DEFLATED) as zs:
        for root, _, files in os.walk(out):
            for f in files:
                p = os.path.join(root, f)
                arc = os.path.join("results_v3", os.path.relpath(p, out))
                (zs if os.sep + "scores" + os.sep in p else zr).write(p, arc)
    for z in (zip_path, scores_zip):
        print(f"{z}: {os.path.getsize(z) / 1e6:.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--zip", default="results_v3.zip")
    ap.add_argument("--budget-h", type=float, default=10.0,
                    help="stop starting new datasets after this many hours (Kaggle stops at 12)")
    ap.add_argument("--events", type=int, nargs="*", help="test on a few event ids only")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--detectors", choices=["main", "lstm"], default="main",
                    help="main = run A; lstm = run B (LSTM AE + timing, no case studies)")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    bench = os.path.join(a.out, "benchmark")
    tables = os.path.join(a.out, "tables")
    failures = []
    t0 = time.time()
    try:
        extra = ["--events", *map(str, a.events)] if a.events else []
        step("benchmark (95 datasets)",
             [os.path.join(here, "run_benchmark.py"), "--data-dir", a.data_dir, "--out", bench,
              "--budget-h", str(a.budget_h), "--epochs", str(a.epochs),
              "--detectors", a.detectors, *extra], failures)
        for farm, eid, cross in (CASES if a.detectors == "main" else []):
            if a.events and eid not in a.events:
                continue
            case = os.path.join(a.out, f"case_{eid}")
            if not os.path.exists(os.path.join(case, f"summary_{eid}.json")):
                step(f"case study {farm}-{eid}",
                     [os.path.join(here, "analyze_case.py"), "--data-dir", a.data_dir,
                      "--farm", farm, "--event", str(eid), "--cross", str(cross),
                      "--epochs", str(a.epochs), "--out", case], failures)
        step("result tables", [os.path.join(here, "merge_results.py"), "--inputs", bench,
                              "--out", tables], failures)
    finally:
        with open(os.path.join(a.out, "RUN_INFO.txt"), "w") as f:
            f.write(f"total time: {(time.time() - t0) / 3600:.2f} h\n")
            f.write(f"failed steps: {', '.join(failures) or 'none'}\n")
        make_zips(a.out, a.zip)
    print(f"\nAll done in {(time.time() - t0) / 3600:.2f} h. Failed steps: {failures or 'none'}")
    print(f"Download {a.zip}")


if __name__ == "__main__":
    main()
