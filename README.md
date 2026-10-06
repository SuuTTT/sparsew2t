# SparseWarn2Trade (label-scarcity track)

Anchored weak supervision for market-manipulation detection with enforcement cases published by decision time.
Plan: docs/sparse_label_research_plan.md. All compute runs as cq jobs from the project home
/home/sudingli/cq-home/sparsew2t/{repo,env,data,results,cache} (GitHub workflow: cq clones SuuTTT/sparsew2t into the home on c224; CPU-only jobs); specs in cq/.

    python tests/test_smoke.py                                  # 12 tests
    python scripts/run_experiment.py --config configs/smoke.yaml --override experiment.device=cpu
