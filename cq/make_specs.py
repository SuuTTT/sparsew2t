#!/usr/bin/env python3
"""Write cq job specs (one per run) for the SparseWarn2Trade project home. Nothing is submitted here.

    python cq/make_specs.py --commit <sha> --tag syn   --configs configs/default.yaml configs/baselines/*.yaml --k 1 2 3 5 10 full
    python cq/make_specs.py --commit <sha> --tag crypto --dataset-config configs/datasets/crypto_pnd_fast.yaml --configs ... --ram-gb 12
    cq submit cq/specs/<tag>/<file>.json      (smoke first; batch specs carry depends_on once the smoke id is known)
"""
import argparse, glob, json, os

HOME = "/home/sudingli/cq-home/sparsew2t"


def spec(name, cmd, commit, ram_gb, cpus, hours, dest, depends_on=None, data_gb=2):
    s = {"name": name, "work": "sparsew2t",
         "command": ["bash", "-lc", cmd],
         "python": f"{HOME}/env/bin/python",
         "code": {"git": "SuuTTT/sparsew2t", "commit": commit},
         "data": [{"path": f"{HOME}/data", "gb": data_gb}],
         "home": {"name": "sparsew2t", "from": "c224"},
         "env": {"OMP_NUM_THREADS": str(cpus), "MKL_NUM_THREADS": str(cpus), "PYTHONUNBUFFERED": "1", "LD_LIBRARY_PATH": f"{HOME}/env/lib"},
         "resources": {"gpus": 0, "cpus": cpus, "ram_gb": ram_gb},
         "limits": {"max_hours": hours, "max_attempts": 2},
         "outputs": {"dest": dest, "expect": [{"path": "DONE"}, {"path": "rows.csv", "min_lines": 10}],
                     "progress": {"path": "progress.txt", "stall_min": 90}},
         "workload": {"bound": "cpu"}}
    if depends_on:
        s["depends_on"] = depends_on if isinstance(depends_on, list) else [depends_on]
    return s


def run_cmd(cfg, dcfg, seed, k, protocols):
    d = f" --dataset-config {dcfg}" if dcfg else ""
    return (f"ln -sfn {HOME}/data data && {HOME}/env/bin/python -W ignore scripts/run_experiment.py --config {cfg}{d} --seeds {seed} --k {' '.join(k)} "
            f"--protocols {' '.join(protocols)} --override experiment.device=cpu --out $CQ_OUTPUT_DIR/run > $CQ_OUTPUT_DIR/run.log 2>&1 "
            f"&& cp $(ls -d $CQ_OUTPUT_DIR/run/*/*/ | head -1)*.csv $CQ_OUTPUT_DIR/ && touch $CQ_OUTPUT_DIR/DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--dataset-config", default=None); ap.add_argument("--seeds", nargs="+", default=["42", "2024", "2025", "2026"])
    ap.add_argument("--k", nargs="+", default=["1", "2", "3", "5", "10", "full"]); ap.add_argument("--protocols", nargs="+", default=["release"])
    ap.add_argument("--ram-gb", type=float, default=6); ap.add_argument("--cpus", type=int, default=4); ap.add_argument("--hours", type=float, default=3)
    ap.add_argument("--depends-on", default=None)
    a = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs", a.tag); os.makedirs(out, exist_ok=True)
    n = 0
    for c in [c for p in a.configs for c in sorted(glob.glob(p))]:
        base = os.path.splitext(os.path.basename(c))[0]
        label = ("bl_" if "/baselines/" in c else "ab_" if "/ablations/" in c else "") + base
        for s in a.seeds:
            dest = f"{HOME}/results/{a.tag}/{label}/seed{s}"
            js = spec(f"sparsew2t {a.tag} {label} seed{s}", run_cmd(c, a.dataset_config, s, a.k, a.protocols), a.commit,
                      a.ram_gb, a.cpus, a.hours, dest, a.depends_on)
            json.dump(js, open(os.path.join(out, f"{label}_seed{s}.json"), "w"), indent=1); n += 1
    print(n, "specs ->", out)


if __name__ == "__main__":
    main()
