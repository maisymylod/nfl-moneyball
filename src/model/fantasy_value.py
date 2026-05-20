"""Compute half-PPR fantasy-draft surplus value for the in-scope window.

Joins each ADP player against per-season fantasy point totals computed from
the seasonal stats raw file, then fits a per-position log-linear baseline of
realized points vs ADP and reports surplus = realized minus expected.

Defensibility rule mirrors the contracts pipeline: ADP players who don't
appear in the season's stats (missed the year, injured, never on a roster)
keep NaN realized + NaN surplus rather than being imputed to zero.

Outputs: data/processed/fantasy_value.parquet (one row per (season, player)).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from src.pipeline.merge import normalize_name

RAW = Path("data/raw")
PROC = Path("data/processed")
FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")
ADP_TEAMS = 12  # rounds derived as ceil(adp / ADP_TEAMS)

log = logging.getLogger(__name__)


def _half_ppr_points(seasonal: pd.DataFrame) -> pd.Series:
    """fantasy_points (standard) + 0.5 * receptions."""
    base = pd.to_numeric(seasonal.get("fantasy_points"), errors="coerce")
    recs = pd.to_numeric(seasonal.get("receptions"), errors="coerce").fillna(0)
    return base + 0.5 * recs


def _build_realized() -> pd.DataFrame:
    """For each (player_id, season), the half-PPR points and name/position
    to support an ID-then-name fallback join."""
    seasonal = pd.read_parquet(RAW / "seasonal.parquet")
    rosters = pd.read_parquet(RAW / "rosters.parquet")

    seasonal = seasonal.copy()
    seasonal["realized_points"] = _half_ppr_points(seasonal)
    seasonal = seasonal[["player_id", "season", "realized_points"]]

    rosters = rosters[rosters["position"].isin(FANTASY_POSITIONS)].copy()
    keep = ["player_id", "season", "player_name", "position", "team"]
    rosters = rosters[[c for c in keep if c in rosters.columns]]

    df = rosters.merge(seasonal, on=["player_id", "season"], how="left")
    df["name_norm"] = df["player_name"].apply(normalize_name)
    return df


def _fit_position_baseline(adp: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    mask = ~np.isnan(points)
    if mask.sum() < 5:
        return None
    x = np.log(adp[mask]).reshape(-1, 1)
    y = points[mask]
    model = HuberRegressor(max_iter=200)
    model.fit(x, y)
    return np.array([model.intercept_, model.coef_[0]])


def _baseline_predict(coefs: np.ndarray | None, adp: np.ndarray) -> np.ndarray:
    if coefs is None:
        return np.full(adp.shape, np.nan)
    return coefs[0] + coefs[1] * np.log(adp)


def _join_realized(adp_row: pd.Series, realized: pd.DataFrame) -> tuple[float, str | None, str | None]:
    """Try team-aware exact-name match, then any-team exact, return (points, player_id, matched_team).
    Returns (NaN, None, None) if no match — the defensibility rule says we never fabricate."""
    season = adp_row["season"]
    pos = adp_row["position"]
    name_norm = normalize_name(adp_row["player_name"])
    if not name_norm:
        return (np.nan, None, None)
    cand = realized[(realized["season"] == season) & (realized["position"] == pos) & (realized["name_norm"] == name_norm)]
    if cand.empty:
        return (np.nan, None, None)
    if (cand["team"] == adp_row.get("team")).any():
        cand = cand[cand["team"] == adp_row.get("team")]
    row = cand.iloc[0]
    return (float(row["realized_points"]) if pd.notna(row["realized_points"]) else np.nan,
            row["player_id"],
            row["team"])


def compute() -> None:
    PROC.mkdir(parents=True, exist_ok=True)

    adp = pd.read_parquet(RAW / "adp.parquet")
    if adp.empty:
        log.warning("no ADP data; skipping")
        return
    adp = adp[adp["position"].isin(FANTASY_POSITIONS)].copy()

    realized = _build_realized()

    realized_pts, player_ids, matched_teams = [], [], []
    for _, row in adp.iterrows():
        pts, pid, team = _join_realized(row, realized)
        realized_pts.append(pts)
        player_ids.append(pid)
        matched_teams.append(team)
    adp["realized_points"] = realized_pts
    adp["player_id"] = player_ids
    adp["matched_team"] = matched_teams
    adp["has_stats"] = adp["realized_points"].notna()

    # Per-position baseline: realized_points ~ a + b * log(ADP)
    adp["expected_points"] = np.nan
    for pos in FANTASY_POSITIONS:
        mask = adp["position"] == pos
        if not mask.any():
            continue
        coefs = _fit_position_baseline(
            adp.loc[mask, "adp"].to_numpy(dtype=float),
            adp.loc[mask, "realized_points"].to_numpy(dtype=float),
        )
        adp.loc[mask, "expected_points"] = _baseline_predict(
            coefs, adp.loc[mask, "adp"].to_numpy(dtype=float)
        )

    # Surplus stays NaN when realized is NaN — no fabrication.
    adp["surplus"] = adp["realized_points"] - adp["expected_points"]
    adp["adp_round"] = np.ceil(adp["adp"] / ADP_TEAMS).astype("Int64")

    keep = [
        "season",
        "adp",
        "adp_round",
        "player_name",
        "position",
        "team",
        "matched_team",
        "player_id",
        "ffc_player_id",
        "times_drafted",
        "high",
        "low",
        "stdev",
        "bye",
        "realized_points",
        "expected_points",
        "surplus",
        "has_stats",
        "adp_format",
        "adp_teams",
    ]
    keep = [c for c in keep if c in adp.columns]
    out = adp[keep].sort_values(["season", "adp"]).reset_index(drop=True)
    out.to_parquet(PROC / "fantasy_value.parquet", index=False)
    log.info("fantasy value: %d player-seasons -> %s", len(out), PROC / "fantasy_value.parquet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    compute()
