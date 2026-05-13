"""Merge raw stats, snaps, rosters, and contracts into a single player-season table.

Implements the defensibility logic the LinkedIn post called out:
  - missing contract values stay NaN (never imputed to 0)
  - every row carries a contract_match_quality flag
  - low-sample players are flagged (not deleted) so they can be audited
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

RAW = Path("data/raw")
PROC = Path("data/processed")
DIAG = Path("data/diagnostic")

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
LOW_SAMPLE_SNAPS = 100

EXACT_MIN = 100
FUZZY_HIGH_MIN = 92
FUZZY_LOW_MIN = 80

log = logging.getLogger(__name__)


def normalize_name(name) -> str:
    if pd.isna(name):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = s.encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    s = re.sub(r"[^a-z\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _join_stats_rosters() -> pd.DataFrame:
    seasonal = pd.read_parquet(RAW / "seasonal.parquet")
    rosters = pd.read_parquet(RAW / "rosters.parquet")
    df = seasonal.merge(rosters, on=["player_id", "season"], how="inner", suffixes=("", "_r"))
    df = df[df["position"].isin(SKILL_POSITIONS)].copy()
    df["name_norm"] = df["player_name"].apply(normalize_name)
    return df.reset_index(drop=True)


def _attach_snaps(df: pd.DataFrame) -> pd.DataFrame:
    snaps_path = RAW / "snaps.parquet"
    if not snaps_path.exists():
        df["offense_snaps"] = pd.NA
        return df
    snaps = pd.read_parquet(snaps_path)
    if snaps.empty or "offense_snaps" not in snaps.columns:
        df["offense_snaps"] = pd.NA
        return df
    snaps = snaps.copy()
    if "player" in snaps.columns:
        snaps["name_norm"] = snaps["player"].apply(normalize_name)
    else:
        snaps["name_norm"] = ""
    agg = (
        snaps.groupby(["name_norm", "season"], dropna=False)["offense_snaps"]
        .sum()
        .reset_index()
    )
    return df.merge(agg, on=["name_norm", "season"], how="left")


def _match_quality(score: float) -> str:
    if score >= EXACT_MIN:
        return "exact"
    if score >= FUZZY_HIGH_MIN:
        return "fuzzy_high"
    if score >= FUZZY_LOW_MIN:
        return "fuzzy_low"
    return "no_match"


def _attach_contracts(df: pd.DataFrame, season: int) -> pd.DataFrame:
    contracts_path = RAW / "contracts.parquet"
    if not contracts_path.exists() or pd.read_parquet(contracts_path).empty:
        log.warning("no contracts available; all rows will be no_match")
        df["apy"] = pd.NA
        df["cap_hit"] = pd.NA
        df["contract_source"] = pd.NA
        df["contract_match_quality"] = "no_match"
        df["contract_match_score"] = 0
        return df

    contracts = pd.read_parquet(contracts_path)
    contracts = contracts[contracts["position"].isin(SKILL_POSITIONS)].copy()
    contracts["name_norm"] = contracts["player_name"].apply(normalize_name)
    contracts = contracts[contracts["name_norm"] != ""]
    contracts = contracts.drop_duplicates(subset=["name_norm", "position"], keep="first")
    by_position = {pos: g for pos, g in contracts.groupby("position")}

    apy_col, cap_col, src_col, qual_col, score_col = [], [], [], [], []
    for _, row in df.iterrows():
        if row["season"] != season:
            apy_col.append(pd.NA)
            cap_col.append(pd.NA)
            src_col.append(pd.NA)
            qual_col.append("not_scored")
            score_col.append(0)
            continue
        cand = by_position.get(row["position"])
        target = row["name_norm"]
        if cand is None or cand.empty or not target:
            apy_col.append(pd.NA)
            cap_col.append(pd.NA)
            src_col.append(pd.NA)
            qual_col.append("no_match")
            score_col.append(0)
            continue
        choices = cand["name_norm"].tolist()
        result = process.extractOne(target, choices, scorer=fuzz.WRatio)
        if result is None:
            apy_col.append(pd.NA)
            cap_col.append(pd.NA)
            src_col.append(pd.NA)
            qual_col.append("no_match")
            score_col.append(0)
            continue
        match_name, score = result[0], result[1]
        quality = _match_quality(score)
        if quality == "no_match":
            apy_col.append(pd.NA)
            cap_col.append(pd.NA)
            src_col.append(pd.NA)
        else:
            matched = cand[cand["name_norm"] == match_name].iloc[0]
            apy_col.append(matched.get("apy", pd.NA))
            cap_col.append(matched.get("cap_hit", pd.NA))
            src_col.append(matched.get("contract_source", pd.NA))
        qual_col.append(quality)
        score_col.append(score)

    df["apy"] = apy_col
    df["cap_hit"] = cap_col
    df["contract_source"] = src_col
    df["contract_match_quality"] = qual_col
    df["contract_match_score"] = score_col
    return df


def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    snaps = df.get("offense_snaps")
    if snaps is None:
        df["low_sample"] = False
    else:
        df["low_sample"] = snaps.fillna(0).astype(float).lt(LOW_SAMPLE_SNAPS)
    return df


def _write_excluded(df: pd.DataFrame, season: int) -> None:
    DIAG.mkdir(parents=True, exist_ok=True)
    current = df[df["season"] == season].copy()
    excluded = current[
        current["low_sample"]
        | current["contract_match_quality"].isin(["no_match", "fuzzy_low"])
    ].copy()

    def reason(r):
        if r["low_sample"]:
            return "low_sample"
        if r["contract_match_quality"] == "no_match":
            return "no_contract_match"
        return "fuzzy_match_below_threshold"

    excluded["exclusion_reason"] = excluded.apply(reason, axis=1)
    cols = [
        c
        for c in (
            "season",
            "player_id",
            "player_name",
            "position",
            "team",
            "offense_snaps",
            "contract_match_quality",
            "contract_match_score",
            "exclusion_reason",
        )
        if c in excluded.columns
    ]
    excluded[cols].sort_values(["exclusion_reason", "player_name"]).to_csv(
        DIAG / "excluded_players.csv", index=False
    )
    log.info("excluded players: %d rows -> %s", len(excluded), DIAG / "excluded_players.csv")


def merge() -> None:
    PROC.mkdir(parents=True, exist_ok=True)

    df = _join_stats_rosters()
    df = _attach_snaps(df)
    current_season = int(df["season"].max())
    df = _attach_contracts(df, current_season)
    df = _add_flags(df)

    out = PROC / "players.parquet"
    df.to_parquet(out, index=False)
    log.info("merged players: %d rows -> %s", len(df), out)

    _write_excluded(df, current_season)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    merge()
