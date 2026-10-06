"""Shared setup for the label-scarcity scripts: config (+ dataset overlay), panel, anchors, LFs."""
from __future__ import annotations
import os
import sys
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from warn2trade.utils import load_config, deep_merge, apply_overrides, DATASETS
from warn2trade.data.anchors import load_anchor_table, synthetic_anchor_table
from warn2trade.data.labels import abnormal_move_labels, impact_curves, bars_to_peak_from_labels
from warn2trade.data.weak_labelers import labeling_functions_from_config
import warn2trade.data.regulatory  # noqa: F401  registers "daily_bars"
import warn2trade.baselines  # noqa: F401  registers SEMI_BASELINES


def load_cfg(config: str, dataset_config: Optional[str], overrides) -> Dict[str, Any]:
    cfg = load_config(config)
    if dataset_config:
        cfg = deep_merge(cfg, load_config(dataset_config))
    return apply_overrides(cfg, overrides)


def build(cfg: Dict[str, Any]) -> Tuple[Any, Any, pd.DataFrame, list]:
    dcfg = dict(cfg["dataset"])
    builder = DATASETS.get(dcfg["name"])(dcfg)
    panel = builder.build_panel()
    acfg = dcfg.get("anchors", {}) or {}
    if acfg.get("path"):
        table = load_anchor_table(acfg["path"])
    elif dcfg["name"] == "synthetic":
        table = synthetic_anchor_table(panel, np.random.default_rng(int(dcfg.get("seed", 0)) + 7), tuple(acfg.get("release_lag", [120, 500])))
        if acfg.get("proxy_labels", "rule") == "rule":
            apply_rule_proxy_labels(panel, builder.horizons)
    else:
        raise ValueError("dataset.anchors.path is required for non-synthetic datasets (build it with scripts/build_anchors.py)")
    if cfg["model"].get("horizons") is None:
        cfg["model"]["horizons"] = list(builder.horizons)
    panel.meta["bars_per_year"] = getattr(builder, "bars_per_year", panel.meta.get("bars_per_year", 252))
    return builder, panel, table, labeling_functions_from_config(cfg["labeling_functions"])


def apply_rule_proxy_labels(panel, horizons) -> None:
    """Synthetic only: Panel.labels := rule labels (what a real builder provides), so that evaluating on anchors is a
    different target from the proxy the weak supervision imitates (guards against circular evaluation)."""
    vol = panel.meta.get("volume")
    panel.labels = abnormal_move_labels(panel.mid, vol if vol is not None else np.ones_like(panel.mid))
    panel.bars_to_peak = bars_to_peak_from_labels(panel.labels, impact_curves(panel.mid, horizons), horizons)
    panel.meta["label_kind"] = "rule_proxy"
