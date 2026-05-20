"""Compute draft-pick surplus value for the in-scope window.

Joins each drafted skill-position player against the current value rankings,
fits a per-position baseline of realized production-residual vs log(pick), and
reports surplus = realized minus baseline. Players without rankings (cut,
retired, injured, never qualified) keep NaN realized + NaN surplus — same
defensibility rule as the contracts pipeline.

Outputs:
  data/processed/draft_value.parquet      one row per drafted player
  data/processed/draft_team_summary.parquet  total/median surplus by team
  data/processed/draft_round_summary.parquet median surplus by round
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

RAW = Path("data/raw")
PROC = Path("data/processed")
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

log = logging.getLogger(__name__)


def _fit_position_baseline(picks: np.ndarray, residuals: np.ndarray) -> np.ndarray | None:
    """Robust log-linear fit of residual ~ log(pick). Returns coefficients [b, m] or None."""
    mask = ~np.isnan(residuals)
    if mask.sum() < 5:
        return None
    x = np.log(picks[mask]).reshape(-1, 1)
    y = residuals[mask]
    model = HuberRegressor(max_iter=200)
    model.fit(x, y)
    return np.array([model.intercept_, model.coef_[0]])


def _baseline_predict(coefs: np.ndarray | None, picks: np.ndarray) -> np.ndarray:
    if coefs is None:
        return np.full(picks.shape, np.nan)
    return coefs[0] + coefs[1] * np.log(picks)


def compute() -> None:
    PROC.mkdir(parents=True, exist_ok=True)

    draft = pd.read_parquet(RAW / "draft.parquet")
    if draft.empty:
        log.warning("no draft data; skipping")
        return

    draft = draft[draft["position"].isin(SKILL_POSITIONS)].copy()
    rankings = pd.read_parquet(PROC / "value_rankings.parquet")
    realized = rankings[
        ["player_id", "production_residual", "value_score", "offense_snaps"]
    ].rename(
        columns={
            "production_residual": "realized_residual",
            "value_score": "realized_value_score",
            "offense_snaps": "current_snaps",
        }
    )

    merged = draft.merge(realized, on="player_id", how="left")
    merged["in_rankings"] = merged["realized_residual"].notna()

    # Per-position baseline: realized_residual ~ log(pick)
    merged["expected_residual"] = np.nan
    for pos in SKILL_POSITIONS:
        mask = merged["position"] == pos
        if not mask.any():
            continue
        coefs = _fit_position_baseline(
            merged.loc[mask, "pick"].to_numpy(dtype=float),
            merged.loc[mask, "realized_residual"].to_numpy(dtype=float),
        )
        merged.loc[mask, "expected_residual"] = _baseline_predict(
            coefs, merged.loc[mask, "pick"].to_numpy(dtype=float)
        )

    # Surplus stays NaN when realized is NaN (no fabrication of bust value).
    merged["surplus"] = merged["realized_residual"] - merged["expected_residual"]

    keep = [
        "season",
        "round",
        "pick",
        "team",
        "player_id",
        "player_name",
        "position",
        "college",
        "age",
        "current_snaps",
        "realized_residual",
        "realized_value_score",
        "expected_residual",
        "surplus",
        "in_rankings",
    ]
    keep = [c for c in keep if c in merged.columns]
    out = merged[keep].sort_values(["season", "pick"]).reset_index(drop=True)
    out.to_parquet(PROC / "draft_value.parquet", index=False)
    log.info("draft value: %d picks -> %s", len(out), PROC / "draft_value.parquet")

    # Team summary: sum of surplus (NaN-safe), count of picks, count of contributors
    team = (
        out.groupby("team", dropna=False)
        .agg(
            picks=("pick", "count"),
            contributors=("in_rankings", "sum"),
            total_surplus=("surplus", "sum"),
            median_surplus=("surplus", "median"),
        )
        .reset_index()
        .sort_values("total_surplus", ascending=False, na_position="last")
    )
    team.to_parquet(PROC / "draft_team_summary.parquet", index=False)
    log.info("draft team summary: %d teams", len(team))

    # Round summary: median surplus (ignores NaN) per round
    rnd = (
        out.groupby("round", dropna=False)
        .agg(
            picks=("pick", "count"),
            contributors=("in_rankings", "sum"),
            median_surplus=("surplus", "median"),
        )
        .reset_index()
        .sort_values("round")
    )
    rnd.to_parquet(PROC / "draft_round_summary.parquet", index=False)
    log.info("draft round summary: %d rounds", len(rnd))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    compute()
