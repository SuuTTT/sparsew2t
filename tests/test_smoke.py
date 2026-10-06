"""Label-scarcity track tests. Run: python3 tests/test_smoke.py   (or python -m pytest tests/test_smoke.py -q)

Covers the fixes from the CCF-A review: evaluation on anchors (not on the proxy labels), nested label budgets, BH-conformal
gate validity and enforcement on teacher positives, held-out calibration, faithful Deep SAD scoring, netted portfolio
accounting, timing-permutation null, event-level bootstrap, Theorem 1 (anchor basin selection), daily-bars builder
(A-share price limits), sheet-to-anchors conversion, and end-to-end runs of the method and a baseline.
"""
from __future__ import annotations
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from warn2trade.utils import load_config, set_seed, DATASETS  # noqa: E402
from warn2trade.data.anchors import synthetic_anchor_table, available_anchors, sample_k_shot  # noqa: E402
from warn2trade.data.weak_labelers import labeling_functions_from_config, build_vote_matrix, lf_report  # noqa: E402
from warn2trade.data.semi_dataset import SemiEventDataset, LabelBudget, make_labeled_arrays, eval_label_arrays  # noqa: E402
from warn2trade.semi.label_model import AnchoredLabelModel, majority_vote_proba, basin_selection_bound, cases_needed  # noqa: E402
from warn2trade.semi.calibration import conformal_pvalues, bh_reject, plca  # noqa: E402
from warn2trade.semi.losses import consistency_loss  # noqa: E402
from warn2trade.backtest.walk_forward import RollingExperiment  # noqa: E402
from warn2trade.backtest.portfolio_eval import netted_backtest, timing_permutation_pvalue  # noqa: E402
from warn2trade.backtest.event_eval import event_level_summary, paired_event_delta  # noqa: E402
from warn2trade.backtest.costs import CostModel  # noqa: E402
from warn2trade.backtest import metrics as M  # noqa: E402
from _sparse_common import apply_rule_proxy_labels  # noqa: E402
import warn2trade.baselines  # noqa: E402,F401
import warn2trade.data.regulatory  # noqa: E402,F401

try:
    import pytest
    parametrize = pytest.mark.parametrize
except ImportError:
    parametrize = lambda name, values: (lambda fn: fn)


def _simulate_votes(n=20000, pi=0.02, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pi).astype(int)
    alpha, beta = np.array([0.9, 0.8, 0.7, 0.65, 0.6]), np.array([0.3, 0.5, 0.4, 0.6, 0.2])
    V = np.zeros((n, 5), dtype=np.int8)
    for j in range(5):
        cov, correct = rng.random(n) < beta[j], rng.random(n) < alpha[j]
        sign = np.where(y == 1, 1, -1)
        V[:, j] = np.where(cov, np.where(correct, sign, -sign), 0)
    return V, y, alpha, beta


def _panel(T=900, N=5, seed=1, rate=0.012, proxy=True):
    b = DATASETS.get("synthetic")({"T": T, "N": N, "seed": seed, "event_rate": rate})
    p = b.build_panel()
    if proxy:
        apply_rule_proxy_labels(p, b.horizons)
    return b, p


# ----------------------------------------------------------------------------- label model and Theorem 1
def test_label_model_anchors_resolve_sign_and_recover_accuracies():
    V, y, alpha, _ = _simulate_votes()
    rng = np.random.default_rng(1)
    y_l = np.full(len(y), -1)
    y_l[rng.choice(np.where(y == 1)[0], 10, replace=False)] = 1
    y_l[rng.choice(np.where(y == 0)[0], 200, replace=False)] = 0
    lm_flip = AnchoredLabelModel(5, prior_pos=0.5, alpha_init=0.3).fit(V, None)
    lm_anch = AnchoredLabelModel(5, prior_pos=0.5, alpha_init=0.3).fit(V, y_l)
    assert (lm_flip.alpha < 0.5).all(), "without anchors EM keeps the mirrored basin"
    assert np.abs(lm_anch.alpha - alpha).max() < 0.08, lm_anch.alpha
    assert M.auc_pr(y, lm_anch.predict_proba(V)) > M.auc_pr(y, majority_vote_proba(V)) - 0.02


def test_theorem1_basin_selection_bound_holds_and_decays():
    rng = np.random.default_rng(0)
    alpha, beta = np.array([0.62, 0.58, 0.55]), np.array([0.5, 0.4, 0.3])
    w = np.log(alpha / (1 - alpha))
    n_rows = 4
    rates, bounds = [], []
    for K in (1, 3, 10, 30):
        wrong = 0
        for _ in range(4000):
            D = 0.0
            for _k in range(K):
                common = rng.random() < 0.5                      # strong within-case dependence: rows share one draw
                for _m in range(n_rows):
                    cov = rng.random(3) < beta
                    agree = (rng.random(3) < alpha) if not common else np.repeat(rng.random() < alpha.mean(), 3)
                    D += float((np.where(agree, 1, -1) * w * cov).sum())
            wrong += D <= 0
        rates.append(wrong / 4000)
        bounds.append(basin_selection_bound(alpha, beta, n_rows, K))
    assert all(r <= b + 0.02 for r, b in zip(rates, bounds)), (rates, bounds)
    assert rates[-1] < rates[0]
    assert cases_needed(alpha, beta, 0.05) >= 1


# ----------------------------------------------------------------------------- conformal + BH gate
def test_conformal_validity_and_bh_fdr():
    rng = np.random.default_rng(0)
    cal = rng.standard_normal(500)
    assert abs((conformal_pvalues(cal, rng.standard_normal(20000)) <= 0.05).mean() - 0.05) < 0.02
    fdrs = []
    for r in range(200):
        cal = rng.standard_normal(400)
        nulls, alts = rng.standard_normal(950), rng.standard_normal(50) + 3.0
        s = np.r_[nulls, alts]
        rej = bh_reject(conformal_pvalues(cal, s), 0.1)
        fdrs.append(rej[:950].sum() / max(rej.sum(), 1))
    assert np.mean(fdrs) <= 0.1 + 0.02, np.mean(fdrs)
    assert plca(np.array([True, True, False]), np.array([1, 0, 1])) == 0.5


def test_gate_blocks_teacher_positives_outside_selection():
    logit = torch.zeros(6)
    p_t = torch.tensor([0.99, 0.99, 0.99, 0.01, 0.01, 0.6])
    gate = torch.tensor([True, False, False, False, True, True])
    _, mask_rate, hard = consistency_loss(logit, p_t, 0.9, None, gate)
    used_pos = ((hard > 0.5) & (p_t >= 0.9) & gate).sum().item()
    assert used_pos == 1 and abs(mask_rate.item() - 3 / 6) < 1e-6      # 1 gated positive + 2 confident negatives


# ----------------------------------------------------------------------------- protocol, budget, evaluation labels
def test_protocols_nest_and_budget_is_nested():
    _, panel = _panel(T=1200, N=6, seed=3, rate=0.015, proxy=False)
    table = synthetic_anchor_table(panel, np.random.default_rng(0), (50, 200))
    t = 800
    rel, nai, orc = (available_anchors(table, t, p) for p in ("release", "naive", "oracle"))
    assert len(rel) <= len(nai) <= len(orc) == len(table)
    prev = set()
    for k in (1, 2, 3, 5):
        cur = set(sample_k_shot(orc, k, seed=7)["case_id"])
        assert prev <= cur, "label budget must be nested"
        prev = cur
    assert set(sample_k_shot(rel, 2, 7)["case_id"]) <= set(sample_k_shot(orc, 50, 7)["case_id"])


def test_evaluation_uses_anchors_not_proxy_and_calibration_is_held_out():
    b, panel = _panel()
    table = synthetic_anchor_table(panel, np.random.default_rng(0), (40, 120))
    ev = eval_label_arrays(panel, table, "anchors", 3)
    assert (ev["y"] != panel.labels).any(), "proxy labels must differ from anchors in this fixture"
    lfs = labeling_functions_from_config(load_config(os.path.join(ROOT, "configs/default.yaml"))["labeling_functions"])
    votes = build_vote_matrix(panel, lfs)
    lab = make_labeled_arrays(panel, table, 500, LabelBudget(k_per_class=2, protocol="oracle"), votes, train_lo=100)
    ds = SemiEventDataset(panel, np.arange(100, 501), 32, b.horizons, votes, lab["y_l"], n_cal=60, cal_block=10, eval_labels=ev)
    cal, labd, unl = set(ds.cal_indices()), set(ds.labeled_indices()), set(ds.unlabeled_indices())
    assert len(cal) > 0 and not (cal & labd) and not (cal & unl)
    it = ds[int(next(iter(cal)))]
    t, i = int(it["t"]), int(it["asset"])
    assert it["y"].item() == ev["y"][t, i] and it["y_l"].item() == -1
    assert lab["budget"]["k_eff_cases"] <= 2 * len(table["anomaly_class"].unique())
    rep = lf_report(votes, [lf.name for lf in lfs], ev["y"])
    assert rep["coverage"].between(0, 1).all()


# ----------------------------------------------------------------------------- economics and statistics
def test_netted_backtest_accounting_and_permutation_null():
    n, N = 30, 2
    r = np.zeros((n, N)); r[11:16, 0] = 0.01
    tgt = np.zeros((n, N)); tgt[9:14, 0] = 0.5                    # decided at 9..13, filled at 10..14, earns 11..15
    liq = np.zeros((n, N, 3)); liq[..., 2] = 1e6
    zero = CostModel(0.0, False, 0.0, 0.0)
    res = netted_backtest(tgt, r, liq, np.full((n, N), 0.01), np.ones((n, N), bool), zero, latency=1, slot_scale=2.0)
    assert abs(res["ret"].sum() - 0.5 * 0.05) < 1e-12
    res_b = netted_backtest(-tgt, r, liq, np.full((n, N), 0.01), np.ones((n, N), bool), zero, 1, borrow_bps_annual=1e4, bars_per_year=1, slot_scale=2.0)
    assert res_b["borrow"].sum() > 0
    pv = timing_permutation_pvalue(tgt, lambda T_: float(netted_backtest(T_, r, liq, np.full((n, N), 0.01), np.ones((n, N), bool), zero, 1, slot_scale=2.0)["ret"].sum()), 50, 0)
    assert 0 < pv["p"] <= 1 and pv["obs"] >= pv["null_mean"]


def test_event_level_bootstrap_and_paired_delta():
    rng = np.random.default_rng(0)
    rows = []
    for seed in (1, 2):
        for e in range(20):
            for k in range(3):
                rows.append({"fold": 0, "seed": seed, "t": 100 * e + k, "asset": e % 4, "score": 5 + rng.random(), "thr_val": 4.0,
                             "y": 1, "event_id": e, "bars_to_peak": 2 - k})
        for j in range(600):
            rows.append({"fold": 0, "seed": seed, "t": 5000 + j, "asset": j % 4, "score": rng.standard_normal(), "thr_val": 4.0,
                         "y": 0, "event_id": -1, "bars_to_peak": np.nan})
    good = pd.DataFrame(rows)
    s = event_level_summary(good, n_boot=50)
    assert s["n_events"] == 20 and s["event_AUROC"] > 0.95 and s["event_AUROC_lo"] <= s["event_AUROC"] <= s["event_AUROC_hi"]
    bad = good.assign(score=rng.standard_normal(len(good)))
    d = paired_event_delta(good, bad, "event_AUROC", n_boot=50)
    assert d["delta"] > 0.1 and d["lo"] > 0, d      # random rows still win ~0.8 via the per-event max


# ----------------------------------------------------------------------------- real-data plumbing
def test_daily_bars_builder_cn_limits_and_sheet_to_anchors():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=320)
    recs = []
    for code in ("600000", "300750", "688981"):
        px = 10.0
        for k, d in enumerate(dates):
            r = 0.1 if (code == "600000" and k == 200) else rng.normal(0, 0.01)
            px *= 1 + r
            recs.append({"date": d.date(), "ticker": code, "open": px, "high": px * 1.01, "low": px * 0.99, "close": px, "volume": 1e6})
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bars.csv")
        pd.DataFrame(recs).to_csv(path, index=False)
        panel = DATASETS.get("daily_bars")({"path": path, "market": "cn", "min_obs": 100}).build_panel()
        i = panel.assets.index("600000")
        assert not panel.tradable[200, i] and panel.tradable[199, i]
        assert panel.price_feat.shape[-1] == 5 and np.isfinite(panel.price_feat).all()
        from warn2trade.data.regulatory.sheet import sheet_to_anchors
        sheet = pd.DataFrame([
            {"case_id": "C1", "source": "CSRC", "ticker": "600000", "anomaly_class": "pump_dump", "onset_date": str(dates[190].date()),
             "end_date": str(dates[205].date()), "release_date": "2030-01-01"},
            {"case_id": "C2", "source": "CSRC", "ticker": "000001", "anomaly_class": "pump_dump", "onset_date": str(dates[10].date()),
             "end_date": str(dates[20].date()), "release_date": str(dates[300].date())}])
        table, rep = sheet_to_anchors(sheet, panel.times, panel.assets)
        assert len(table) == 1 and int(table["release_t"].iloc[0]) == panel.T, "late release must not be clipped into the panel"
        assert rep["match_rate"].iloc[0] == 0.5


# ----------------------------------------------------------------------------- end to end
@parametrize("variant", ["method", "deep_sad", "baseline"])
def test_end_to_end_rolling(variant="method"):
    set_seed(42)
    cfg = load_config(os.path.join(ROOT, "configs/smoke.yaml"))
    if variant == "deep_sad":
        ds_cfg = load_config(os.path.join(ROOT, "configs/baselines/deep_sad.yaml"))
        for key in ("semi", "loss", "eval"):
            cfg[key].update(ds_cfg[key])
        cfg["model"]["ablation"] = ds_cfg["model"]["ablation"]
    if variant == "baseline":
        cfg["baseline"], cfg["baseline_cfg"] = "label_model_only", {}
    b, panel = _panel(T=1100, N=6, seed=0, rate=0.012)
    cfg["model"]["horizons"] = list(b.horizons)
    table = synthetic_anchor_table(panel, np.random.default_rng(7), (40, 120))
    exp = RollingExperiment(panel, table, cfg, labeling_functions_from_config(cfg["labeling_functions"]))
    rows, scores = exp.run([42], [1, None], ["release"])
    present = set(rows["metric"])
    for m in ("AP", "AUROC", "K_eff", "n_test_events", "net_SR", "ovl_IR", "ovl_IR_perm_p", "net_DSR", "WAD_mean"):
        assert m in present, m
    ev_y = exp.eval_arrays["y"][scores["t"].to_numpy(), scores["asset"].to_numpy()]
    assert (scores["y"].to_numpy() == ev_y).all(), "evaluation must use the anchor raster"
    assert scores["score"].std() > 0, "detector scores must not be constant"
    if variant == "method":
        assert "policy" in set(rows["rule"]) and "gate_FDR_emp" in present
    s = event_level_summary(scores[scores.K == "Full"], n_boot=20)
    assert s.get("n_events", 0) >= 1


if __name__ == "__main__":
    import time
    import traceback
    import warnings
    warnings.filterwarnings("ignore")
    cases = [(n, f, a) for n, f, a in [
        ("label_model_anchors", test_label_model_anchors_resolve_sign_and_recover_accuracies, ()),
        ("theorem1_basin_bound", test_theorem1_basin_selection_bound_holds_and_decays, ()),
        ("conformal_bh_fdr", test_conformal_validity_and_bh_fdr, ()),
        ("gate_on_teacher_positives", test_gate_blocks_teacher_positives_outside_selection, ()),
        ("protocols_nested_budget", test_protocols_nest_and_budget_is_nested, ()),
        ("eval_on_anchors_heldout_cal", test_evaluation_uses_anchors_not_proxy_and_calibration_is_held_out, ()),
        ("netted_backtest_null", test_netted_backtest_accounting_and_permutation_null, ()),
        ("event_bootstrap", test_event_level_bootstrap_and_paired_delta, ()),
        ("daily_bars_and_sheet", test_daily_bars_builder_cn_limits_and_sheet_to_anchors, ()),
        ("e2e_method", test_end_to_end_rolling, ("method",)),
        ("e2e_deep_sad", test_end_to_end_rolling, ("deep_sad",)),
        ("e2e_score_baseline", test_end_to_end_rolling, ("baseline",))]]
    n_ok = 0
    for name, fn, args in cases:
        t0 = time.time()
        try:
            fn(*args); n_ok += 1; print(f"PASS {name} ({time.time() - t0:.1f}s)")
        except Exception:
            print(f"FAIL {name} ({time.time() - t0:.1f}s)"); traceback.print_exc(limit=10)
    print(f"{n_ok}/{len(cases)} passed")
    sys.exit(0 if n_ok == len(cases) else 1)
