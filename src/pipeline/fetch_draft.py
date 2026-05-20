"""Pull NFL draft picks for the in-scope window via nfl_data_py.

Writes data/raw/draft.parquet with one row per drafted player. The downstream
draft-value step joins this against our current value rankings to compute
surplus value at each pick slot.
"""
from __future__ import annotations

import logging
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

RAW_DIR = Path("data/raw")
DRAFT_YEARS = (2021, 2022, 2023, 2024, 2025)

KEEP_COLS = [
    "season",
    "round",
    "pick",
    "team",
    "gsis_id",
    "pfr_player_name",
    "position",
    "category",
    "college",
    "age",
]

log = logging.getLogger(__name__)


def _fetch_per_year(years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        try:
            df = nfl.import_draft_picks([y])
            frames.append(df)
        except Exception as e:
            log.warning("draft picks for %d unavailable: %s", y, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch(years: list[int] | None = None) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    years = list(years) if years else list(DRAFT_YEARS)
    log.info("pulling draft years: %s", years)
    df = _fetch_per_year(years)
    if df.empty:
        log.warning("no draft data fetched")
        df.to_parquet(RAW_DIR / "draft.parquet", index=False)
        return
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].copy()
    df = df.rename(columns={"gsis_id": "player_id", "pfr_player_name": "player_name"})
    df.to_parquet(RAW_DIR / "draft.parquet", index=False)
    log.info("draft picks: %d rows -> %s", len(df), RAW_DIR / "draft.parquet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch()
