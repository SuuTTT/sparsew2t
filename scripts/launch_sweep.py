#!/usr/bin/env python3
"""Run a sweep as independent jobs (config x seed) with N parallel workers; resumable (skips jobs whose output has DONE).

    python3 scripts/launch_sweep.py --tag syn --configs configs/default.yaml configs/baselines/*.yaml --seeds 42 2024 2025 2026 \
        --k 1 2 3 5 10 full --parallel 10
Every job: run_experiment.py --config C [--dataset-config D] --seeds s --k ... --protocols ... --out runs/<tag>/<cfg>/seed<s>
"""
import argparse, glob, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--dataset-config", default=None); ap.add_argument("--seeds", nargs="+", default=["42", "2024", "2025", "2026"])
    ap.add_argument("--k", nargs="+", default=None); ap.add_argument("--protocols", nargs="+", default=["release"])
    ap.add_argument("--override", nargs="*", default=[]); ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    cfgs = [c for pat in a.configs for c in sorted(glob.glob(pat))]
    jobs = []
    for c in cfgs:
        name = os.path.splitext(os.path.basename(c))[0] if "baselines" not in c and "ablations" not in c else \
            ("bl_" if "baselines" in c else "ab_") + os.path.splitext(os.path.basename(c))[0]
        for s in a.seeds:
            out = os.path.join(ROOT, "runs", a.tag, name, f"seed{s}")
            if glob.glob(os.path.join(out, "*", "*", "DONE")):
                continue
            cmd = [a.python, "-W", "ignore", os.path.join(ROOT, "scripts/run_experiment.py"), "--config", c, "--seeds", s, "--out", out,
                   "--protocols", *a.protocols]
            if a.dataset_config:
                cmd += ["--dataset-config", a.dataset_config]
            if a.k:
                cmd += ["--k", *a.k]
            if a.override:
                cmd += ["--override", *a.override]
            jobs.append((name, s, out, cmd))
    print(f"{len(jobs)} jobs ({len(cfgs)} configs x {len(a.seeds)} seeds, finished ones skipped)", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
    t0 = time.time(); state = {"done": 0, "fail": 0}

    def run(job):
        name, s, out, cmd = job
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "job.log"), "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        state["done" if rc == 0 else "fail"] += 1
        print(f"[{time.time() - t0:7.0f}s] {'OK  ' if rc == 0 else 'FAIL'} {name} seed{s} ({state['done']} ok, {state['fail']} failed, {len(jobs)} total)", flush=True)

    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
