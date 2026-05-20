"""Defensibility checks for the fantasy-draft surplus computation.

Mirrors the contracts pipeline rule: an ADP player whose season had no stats
must keep NaN realized + NaN surplus, never 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.model.fantasy_value import (
    _baseline_predict,
    _fit_position_baseline,
    _half_ppr_points,
)


def test_half_ppr_adds_half_per_reception():
    df = pd.DataFrame({"fantasy_points": [100.0, 50.0, np.nan], "receptions": [20.0, 5.0, 8.0]})
    pts = _half_ppr_points(df)
    # 100 + 0.5*20 = 110, 50 + 0.5*5 = 52.5, NaN + 0.5*8 stays NaN
    assert pts.iloc[0] == 110.0
    assert pts.iloc[1] == 52.5
    assert pd.isna(pts.iloc[2])


def test_baseline_decreases_with_adp():
    """Realized points should fall as ADP grows, so the log-linear coefficient
    must come out negative."""
    rng = np.random.default_rng(7)
    adp = np.arange(1, 180, dtype=float)
    points = 320 - 50 * np.log(adp) + rng.normal(0, 10, size=adp.shape)
    coefs = _fit_position_baseline(adp, points)
    assert coefs is not None
    assert coefs[1] < 0
    preds = _baseline_predict(coefs, adp)
    assert preds[0] > preds[-1]


def test_baseline_returns_none_when_too_few_samples():
    assert _fit_position_baseline(np.array([1.0, 50.0]), np.array([5.0, np.nan])) is None


def test_baseline_predict_handles_none_coefs():
    out = _baseline_predict(None, np.array([10.0, 20.0]))
    assert np.all(np.isnan(out))


def test_missing_realized_stays_nan(tmp_path, monkeypatch):
    """End-to-end: an ADP entry with no matching season stats must come out
    with NaN realized and NaN surplus, never 0."""
    from src.model import fantasy_value as fv_mod

    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir()
    proc_dir.mkdir()

    monkeypatch.setattr(fv_mod, "RAW", raw_dir)
    monkeypatch.setattr(fv_mod, "PROC", proc_dir)

    # 8 WR + 1 ghost (only in ADP, never in rosters/seasonal)
    n = 8
    adp = pd.DataFrame(
        {
            "season": [2024] * (n + 1),
            "ffc_player_id": list(range(n + 1)),
            "player_name": [f"WR Player {i}" for i in range(n)] + ["Ghost Receiver"],
            "position": ["WR"] * (n + 1),
            "team": ["LAR"] * (n + 1),
            "adp": [5.0, 15.0, 28.0, 45.0, 60.0, 85.0, 110.0, 140.0, 170.0],
            "times_drafted": [50] * (n + 1),
            "high": [1] * (n + 1),
            "low": [200] * (n + 1),
            "stdev": [5.0] * (n + 1),
            "bye": [10] * (n + 1),
            "adp_format": ["half-ppr"] * (n + 1),
            "adp_teams": [12] * (n + 1),
        }
    )
    adp.to_parquet(raw_dir / "adp.parquet", index=False)

    rosters = pd.DataFrame(
        {
            "player_id": [f"P{i}" for i in range(n)],
            "season": [2024] * n,
            "player_name": [f"WR Player {i}" for i in range(n)],
            "position": ["WR"] * n,
            "team": ["LAR"] * n,
        }
    )
    rosters.to_parquet(raw_dir / "rosters.parquet", index=False)

    seasonal = pd.DataFrame(
        {
            "player_id": [f"P{i}" for i in range(n)],
            "season": [2024] * n,
            "fantasy_points": [280.0, 220.0, 170.0, 140.0, 110.0, 85.0, 60.0, 40.0],
            "receptions": [100, 90, 80, 70, 60, 50, 40, 30],
        }
    )
    seasonal.to_parquet(raw_dir / "seasonal.parquet", index=False)

    fv_mod.compute()

    out = pd.read_parquet(proc_dir / "fantasy_value.parquet")
    ghost = out[out["player_name"] == "Ghost Receiver"].iloc[0]
    assert pd.isna(ghost["realized_points"])
    assert pd.isna(ghost["surplus"])
    assert bool(ghost["has_stats"]) is False

    # A matched WR must have a numeric realized and surplus
    p0 = out[out["player_name"] == "WR Player 0"].iloc[0]
    assert not pd.isna(p0["realized_points"])
    assert not pd.isna(p0["surplus"])
    # And the adp_round derives from ceil(adp / 12)
    assert int(p0["adp_round"]) == 1
