"""Defensibility checks for the draft-surplus computation.

Mirrors the contracts pipeline rule: a drafted player who doesn't appear in
the current value rankings must keep NaN realized + NaN surplus, never 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.model.draft_value import _baseline_predict, _fit_position_baseline


def test_baseline_is_decreasing_in_pick():
    """Realized residuals are larger for early picks, so the fitted baseline
    should slope downward as pick number grows."""
    rng = np.random.default_rng(0)
    picks = np.arange(1, 200, dtype=float)
    # synthetic monotone-decreasing-in-log signal + noise
    residuals = 20 - 3 * np.log(picks) + rng.normal(0, 2, size=picks.shape)
    coefs = _fit_position_baseline(picks, residuals)
    assert coefs is not None
    # log coefficient must be negative
    assert coefs[1] < 0
    preds = _baseline_predict(coefs, picks)
    assert preds[0] > preds[-1]


def test_baseline_returns_none_when_too_few_samples():
    coefs = _fit_position_baseline(np.array([1.0, 50.0]), np.array([5.0, np.nan]))
    assert coefs is None


def test_baseline_predict_handles_none_coefs():
    out = _baseline_predict(None, np.array([10.0, 20.0]))
    assert np.all(np.isnan(out))


def test_missing_realized_stays_nan(tmp_path, monkeypatch):
    """End-to-end: a pick whose player_id isn't in value_rankings must come
    out with NaN realized and NaN surplus."""
    from src.model import draft_value as dv_mod

    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir()
    proc_dir.mkdir()

    monkeypatch.setattr(dv_mod, "RAW", raw_dir)
    monkeypatch.setattr(dv_mod, "PROC", proc_dir)

    # 8 WR picks (enough to fit the per-position baseline) plus one bust
    n = 8
    draft = pd.DataFrame(
        {
            "season": [2024] * (n + 1),
            "round": [1, 1, 2, 2, 3, 3, 4, 5, 6],
            "pick": [3, 12, 35, 50, 70, 90, 120, 170, 220],
            "team": ["AAA"] * (n + 1),
            "player_id": [f"P{i}" for i in range(n)] + ["GHOST"],
            "player_name": [f"WR{i}" for i in range(n)] + ["Bust"],
            "position": ["WR"] * (n + 1),
            "college": ["X"] * (n + 1),
            "age": [22] * (n + 1),
        }
    )
    draft.to_parquet(raw_dir / "draft.parquet", index=False)

    rankings = pd.DataFrame(
        {
            "player_id": [f"P{i}" for i in range(n)],
            "production_residual": [15.0, 8.0, 3.0, 1.0, -2.0, 4.0, -5.0, 0.5],
            "value_score": [2.5, 1.2, 0.4, 0.1, -0.3, 0.6, -0.7, 0.05],
            "offense_snaps": [800, 600, 400, 300, 200, 350, 250, 150],
        }
    )
    rankings.to_parquet(proc_dir / "value_rankings.parquet", index=False)

    dv_mod.compute()

    out = pd.read_parquet(proc_dir / "draft_value.parquet")
    ghost = out[out["player_name"] == "Bust"].iloc[0]
    assert pd.isna(ghost["realized_residual"])
    assert pd.isna(ghost["surplus"])
    assert bool(ghost["in_rankings"]) is False

    # Players in rankings must have a numeric surplus (now that the per-position
    # baseline has enough data points to fit).
    p0 = out[out["player_name"] == "WR0"].iloc[0]
    assert not pd.isna(p0["realized_residual"])
    assert not pd.isna(p0["surplus"])
