"""Pull half-PPR ADP for the in-scope years from Fantasy Football Calculator.

FFC's public API is `https://fantasyfootballcalculator.com/api/v1/adp/half-ppr`,
keyed by `year` and `teams`. We use 12-team leagues (the most common format)
and store one row per (season, player, position) with the consensus ADP and
draft volume metadata.

Writes data/raw/adp.parquet.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw")
ADP_YEARS = (2021, 2022, 2023, 2024, 2025)
FORMAT = "half-ppr"
TEAMS = 12
URL = "https://fantasyfootballcalculator.com/api/v1/adp/{format}"

log = logging.getLogger(__name__)


def _fetch_year(year: int) -> pd.DataFrame:
    """Call FFC for one year. FFC's `year` param doesn't index 2025 (their
    most recent dataset is only reachable via the no-param default), so we
    fall back to the default endpoint when the year query errors."""
    params = {"teams": TEAMS, "year": year}
    try:
        r = requests.get(URL.format(format=FORMAT), params=params, timeout=15)
        r.raise_for_status()
        payload = r.json()
        if payload.get("status") != "Success":
            log.info("FFC year=%d returned status=%s; trying default endpoint", year, payload.get("status"))
            r = requests.get(URL.format(format=FORMAT), params={"teams": TEAMS}, timeout=15)
            r.raise_for_status()
            payload = r.json()
            meta_year = (payload.get("meta", {}).get("start_date") or "")[:4]
            if str(year) != meta_year:
                log.warning("FFC default returned %s data, not %d; skipping", meta_year, year)
                return pd.DataFrame()
    except Exception as e:
        log.warning("FFC %s/%d unavailable: %s", FORMAT, year, e)
        return pd.DataFrame()

    players = payload.get("players", [])
    if not players:
        return pd.DataFrame()
    df = pd.DataFrame(players)
    df["season"] = year
    df["adp_format"] = FORMAT
    df["adp_teams"] = TEAMS
    return df


def fetch(years: list[int] | None = None) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    years = list(years) if years else list(ADP_YEARS)
    log.info("pulling ADP years: %s", years)
    frames = []
    for y in years:
        df = _fetch_year(y)
        if not df.empty:
            frames.append(df)
            log.info("ADP %d: %d players", y, len(df))
        time.sleep(0.3)  # gentle to FFC
    if not frames:
        log.warning("no ADP data fetched")
        pd.DataFrame().to_parquet(RAW_DIR / "adp.parquet", index=False)
        return
    out = pd.concat(frames, ignore_index=True)
    keep = [
        c
        for c in (
            "season",
            "player_id",
            "name",
            "position",
            "team",
            "adp",
            "adp_formatted",
            "times_drafted",
            "high",
            "low",
            "stdev",
            "bye",
            "adp_format",
            "adp_teams",
        )
        if c in out.columns
    ]
    out = out[keep].rename(columns={"player_id": "ffc_player_id", "name": "player_name"})
    out.to_parquet(RAW_DIR / "adp.parquet", index=False)
    log.info("ADP total: %d rows -> %s", len(out), RAW_DIR / "adp.parquet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch()
