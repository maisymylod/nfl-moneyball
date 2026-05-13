"""Apply trained per-position models to the most recent season and compute value scores.

Outputs data/processed/value_rankings.parquet with one row per current-season player.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.features import FEATURE_COLS, build_features

PROC = Path("data/processed")
MODELS = Path("models")

HIGH_SNAPS = 500
MEDIUM_SNAPS = 200

log = logging.getLogger(__name__)


def _confidence(row) -> str:
    snaps = row.get("offense_snaps")
    snaps = 0 if pd.isna(snaps) else float(snaps)
    quality = row.get("contract_match_quality") or "no_match"
    prior = row.get("prior_production")
    has_prior = pd.notna(prior)
    if snaps >= HIGH_SNAPS and quality in {"exact", "fuzzy_high"} and has_prior:
        return "High"
    if snaps >= MEDIUM_SNAPS and quality in {"exact", "fuzzy_high"}:
        return "Medium"
    return "Low"


def score() -> None:
    with open(MODELS / "latest.pkl", "rb") as f:
        bundle = pickle.load(f)
    models = bundle["models"]

    raw = pd.read_parquet(PROC / "players.parquet")
    df = build_features(raw)
    current_season = int(df["season"].max())
    cur = df[df["season"] == current_season].copy()

    cur["expected_production"] = np.nan
    for pos, model in models.items():
        mask = cur["position"] == pos
        if not mask.any():
            continue
        X = cur.loc[mask, FEATURE_COLS].apply(pd.to_numeric, errors="coerce").values
        cur.loc[mask, "expected_production"] = model.predict(X)

    cur["production_residual"] = cur["production"] - cur["expected_production"]

    apy_millions = pd.to_numeric(cur["apy"], errors="coerce") / 1_000_000
    cur["apy_millions"] = apy_millions
    # value_score is intentionally NaN when contract is missing — the whole point.
    cur["value_score"] = cur["production_residual"] / apy_millions.where(apy_millions > 0)

    cur["confidence"] = cur.apply(_confidence, axis=1)

    out_cols = [
        c
        for c in (
            "season",
            "player_id",
            "player_name",
            "position",
            "team",
            "age",
            "years_exp",
            "offense_snaps",
            "production",
            "expected_production",
            "production_residual",
            "apy",
            "apy_millions",
            "cap_hit",
            "contract_source",
            "contract_match_quality",
            "contract_match_score",
            "low_sample",
            "value_score",
            "confidence",
        )
        if c in cur.columns
    ]
    rankings = cur[out_cols].sort_values("value_score", ascending=False, na_position="last")
    rankings.to_parquet(PROC / "value_rankings.parquet", index=False)
    log.info("value rankings: %d rows -> %s", len(rankings), PROC / "value_rankings.parquet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    score()
