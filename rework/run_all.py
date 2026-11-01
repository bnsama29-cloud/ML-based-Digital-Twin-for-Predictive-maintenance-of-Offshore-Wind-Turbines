"""Everything in one go, on one Kaggle account:
benchmark on all 95 datasets (farms A, B, C) -> dataset-44 deep dive -> paper tables.

    python run_all.py --data-dir /tmp/care --out /kaggle/working/results

Safe to rerun after a crash: finished datasets are skipped and the deep dive
is skipped if its summary already exists.
"""
import argparse
import os
import subprocess
import sys
import time


def step(name, cmd):
    print(f"\n===== {name} =====", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, *cmd], check=True)
    print(f"===== {name} done in {(time.time() - t0) / 60:.1f} min =====", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--events", type=int, nargs="*", help="test on a few event ids only")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    bench = os.path.join(a.out, "benchmark")
    deep = os.path.join(a.out, "deep44")
    tables = os.path.join(a.out, "paper_tables")

    extra = ["--events", *map(str, a.events)] if a.events else []
    # keep 44 and 61 on disk: the deep dive needs them after the benchmark
    step("benchmark (95 datasets)", [os.path.join(here, "run_benchmark.py"),
                                      "--data-dir", a.data_dir, "--out", bench, *extra])
    if not os.path.exists(os.path.join(deep, "summary_44.json")):
        step("dataset-44 deep dive", [os.path.join(here, "analyze_44.py"),
                                      "--data-dir", a.data_dir, "--out", deep])
    step("paper tables", [os.path.join(here, "merge_results.py"),
                          "--inputs", bench, "--out", tables])
    print(f"\nAll done. Download the folder {a.out}")


if __name__ == "__main__":
    main()
