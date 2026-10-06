# SparseWarn2Trade (label-scarcity track)

Anchored weak supervision for market-manipulation detection with enforcement cases published by decision time.
Plan: docs/sparse_label_research_plan.md. All compute runs as cq jobs from the project home
/home/sudingli/cq-home/sparsew2t/{repo,env,data,results,cache} (home server c227, CPU only); specs in cq/.

    python tests/test_smoke.py                                  # 12 tests
    python scripts/run_experiment.py --config configs/smoke.yaml --override experiment.device=cpu
