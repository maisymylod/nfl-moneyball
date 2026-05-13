"""Per-position feature builders and production metric.

Production metric per position (total EPA contribution):
  QB: passing_epa + rushing_epa
  RB: rushing_epa + receiving_epa
  WR: receiving_epa + rushing_epa
  TE: receiving_epa + rushing_epa

Features for predicting season t production:
  age, years_exp, offense_snaps, prior_production (t-1), prior_snaps (t-1)
"""
from __future__ import annotations

import pandas as pd

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

FEATURE_COLS = [
    "age",
    "years_exp",
    "offense_snaps",
    "prior_production",
    "prior_snaps",
]
TARGET = "production"


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


def compute_production(df: pd.DataFrame) -> pd.Series:
    passing = _num(df, "passing_epa")
    rushing = _num(df, "rushing_epa")
    receiving = _num(df, "receiving_epa")
    position = df["position"]

    prod = pd.Series(0.0, index=df.index)
    is_qb = position == "QB"
    prod = prod.mask(is_qb, passing + rushing)
    is_skill = position.isin(["RB", "WR", "TE"])
    prod = prod.mask(is_skill, rushing + receiving)
    return prod


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["production"] = compute_production(df)
    df = df.sort_values(["player_id", "season"]).reset_index(drop=True)
    df["prior_production"] = df.groupby("player_id")["production"].shift(1)
    df["prior_snaps"] = df.groupby("player_id")["offense_snaps"].shift(1)
    return df
