"""Pull seasonal NFL skill-position stats via nfl_data_py.

Writes three raw parquet files to data/raw/:
  - seasonal.parquet   per-player-season aggregated stats
  - snaps.parquet      per-player-season snap totals
  - rosters.parquet    per-player-season metadata (age, exp, team)
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

RAW_DIR = Path("data/raw")
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

log = logging.getLogger(__name__)


def latest_completed_season() -> int:
    today = dt.date.today()
    if today.month >= 3:
        return today.year - 1
    return today.year - 2


def seasons_to_pull(n: int = 5) -> list[int]:
    end = latest_completed_season()
    return list(range(end - n + 1, end + 1))


def _fetch_per_year(fn, years: list[int], label: str) -> pd.DataFrame:
    """Call a nfl_data_py loader one year at a time, skipping years that 404."""
    frames = []
    for y in years:
        try:
            df = fn([y])
            df["season"] = y if "season" not in df.columns else df["season"]
            frames.append(df)
        except Exception as e:
            log.warning("%s for %d unavailable: %s", label, y, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_seasonal(years: list[int]) -> pd.DataFrame:
    return _fetch_per_year(lambda ys: nfl.import_seasonal_data(ys, s_type="REG"), years, "seasonal")


def _fetch_rosters(years: list[int]) -> pd.DataFrame:
    df = _fetch_per_year(nfl.import_seasonal_rosters, years, "rosters")
    if df.empty:
        return df
    keep = [
        c
        for c in (
            "player_id",
            "player_name",
            "position",
            "team",
            "age",
            "years_exp",
            "season",
            "draft_round",
            "draft_pick",
            "birth_date",
        )
        if c in df.columns
    ]
    df = df[keep].copy()
    df = df[df["position"].isin(SKILL_POSITIONS)]
    return df


def _fetch_snaps(years: list[int]) -> pd.DataFrame:
    weekly = _fetch_per_year(nfl.import_snap_counts, years, "snap_counts")
    if weekly.empty:
        return pd.DataFrame(
            columns=["player", "position", "team", "season", "offense_snaps"]
        )
    if "offense_snaps" not in weekly.columns:
        log.warning("offense_snaps column missing in snap_counts payload")
        return pd.DataFrame(
            columns=["player", "position", "team", "season", "offense_snaps"]
        )
    weekly = weekly[weekly["position"].isin(SKILL_POSITIONS)]
    keys = [k for k in ("player", "position", "team", "season") if k in weekly.columns]
    if not keys:
        return pd.DataFrame(columns=["player", "position", "team", "season", "offense_snaps"])
    return weekly.groupby(keys, dropna=False)["offense_snaps"].sum().reset_index()


def fetch(years: list[int] | None = None) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    years = years or seasons_to_pull()
    log.info("pulling seasons: %s", years)

    seasonal = _fetch_seasonal(years)
    seasonal.to_parquet(RAW_DIR / "seasonal.parquet", index=False)
    log.info("seasonal: %d rows", len(seasonal))

    rosters = _fetch_rosters(years)
    rosters.to_parquet(RAW_DIR / "rosters.parquet", index=False)
    log.info("rosters: %d rows", len(rosters))

    snaps = _fetch_snaps(years)
    snaps.to_parquet(RAW_DIR / "snaps.parquet", index=False)
    log.info("snaps: %d rows", len(snaps))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch()
