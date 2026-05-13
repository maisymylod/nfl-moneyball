"""Pull NFL contract data from OverTheCap by position.

Strategy:
  1. Try to scrape OverTheCap position pages (free, public).
  2. If a position fails, log warning and continue.
  3. If everything fails, fall back to a small bundled seed list of top contracts.

The bundled seed is intentionally small and only used as a last resort so the
pipeline never produces an empty contracts table. The merged player table
always carries a `contract_source` column so downstream consumers can see
exactly where each row came from.
"""
from __future__ import annotations

import logging
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw")
SEED_PATH = Path("data/seed/contracts_seed.csv")

POSITION_URLS = {
    "QB": "https://overthecap.com/position/quarterback",
    "RB": "https://overthecap.com/position/running-back",
    "WR": "https://overthecap.com/position/wide-receiver",
    "TE": "https://overthecap.com/position/tight-end",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

log = logging.getLogger(__name__)


def _parse_money(val) -> float | None:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s in {"-", "--", "N/A"}:
        return None
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", "-", "."}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _scrape_position(position: str, url: str) -> pd.DataFrame:
    log.info("fetching %s contracts from %s", position, url)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise RuntimeError(f"no tables on {url}")
    df = max(tables, key=len).copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    rename = {}
    for c in df.columns:
        if c == "player":
            rename[c] = "player_name"
        elif c in {"team", "tm"}:
            rename[c] = "team"
        elif "avg" in c or c in {"apy", "average per year"}:
            rename[c] = "apy"
        elif "cap" in c and "hit" in c:
            rename[c] = "cap_hit"
        elif c in {"value", "total value", "contract value"}:
            rename[c] = "total_value"
        elif "year" in c and ("length" in c or "yrs" in c or c == "years"):
            rename[c] = "years"
    df = df.rename(columns=rename)

    keep = [c for c in ("player_name", "team", "apy", "cap_hit", "total_value", "years") if c in df.columns]
    if "player_name" not in keep or "apy" not in keep:
        raise RuntimeError(f"missing required columns on {url}: have {df.columns.tolist()}")

    df = df[keep].copy()
    for col in ("apy", "cap_hit", "total_value"):
        if col in df.columns:
            df[col] = df[col].apply(_parse_money)

    df["position"] = position
    df["contract_source"] = "overthecap"
    return df


def _load_seed() -> pd.DataFrame:
    if not SEED_PATH.exists():
        log.warning("no seed file at %s", SEED_PATH)
        return pd.DataFrame(
            columns=["player_name", "team", "apy", "cap_hit", "position", "contract_source"]
        )
    df = pd.read_csv(SEED_PATH)
    df["contract_source"] = "seed_fallback"
    return df


def fetch() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for pos, url in POSITION_URLS.items():
        try:
            frames.append(_scrape_position(pos, url))
            time.sleep(1.5)
        except Exception as e:
            log.warning("scrape failed for %s: %s", pos, e)

    if not frames:
        log.warning("all OTC scrapes failed; using bundled seed")
        df = _load_seed()
    else:
        df = pd.concat(frames, ignore_index=True)

    out = RAW_DIR / "contracts.parquet"
    df.to_parquet(out, index=False)
    log.info("contracts: %d rows -> %s", len(df), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch()
