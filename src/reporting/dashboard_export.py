"""Compile every artifact the static HTML dashboard needs into docs/data.json.

Single fetch URL keeps the front end simple: no parquet parsing in the browser,
no multiple round-trips. The daily cron commits this file along with the other
artifacts, and GitHub Pages republishes automatically.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(".")
PROC = ROOT / "data" / "processed"
DIAG = ROOT / "data" / "diagnostic"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"

PLAYER_COLS = [
    "player_id",
    "player_name",
    "position",
    "team",
    "age",
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
]

log = logging.getLogger(__name__)


def _clean(v):
    """JSON-safe coercion: NaN/inf become None, numpy bools/ints/floats become Python."""
    if v is None:
        return None
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if hasattr(v, "item"):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            return v
        return _clean(v)
    return v


def _records(df: pd.DataFrame, cols: list[str] | None = None) -> list[dict]:
    if cols is not None:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    out = df.copy()
    for c in out.select_dtypes(include="number").columns:
        out[c] = out[c].astype(float).round(3)
    records = out.to_dict(orient="records")
    return [{k: _clean(v) for k, v in r.items()} for r in records]


def _load_movers() -> list[dict]:
    snaps_dir = PROC / "snapshots"
    if not snaps_dir.exists():
        return []
    files = sorted(snaps_dir.glob("*.parquet"))
    if len(files) < 2:
        return []
    cur = pd.read_parquet(files[-1])
    prev = pd.read_parquet(files[-2])
    merged = cur.merge(
        prev[["player_id", "value_score", "production_residual"]],
        on="player_id",
        how="inner",
        suffixes=("", "_prev"),
    )
    merged["value_score_delta"] = merged["value_score"] - merged["value_score_prev"]
    movers = merged.dropna(subset=["value_score_delta"])
    if movers.empty:
        return []
    cols = [
        "player_name",
        "position",
        "team",
        "value_score_prev",
        "value_score",
        "value_score_delta",
        "confidence",
    ]
    return _records(movers.sort_values("value_score_delta", ascending=False), cols)


def _load_history() -> list[dict]:
    path = PROC / "run_history.parquet"
    if not path.exists():
        return []
    h = pd.read_parquet(path)
    return _records(h.sort_values("run_at"))


def _load_excluded() -> list[dict]:
    path = DIAG / "excluded_players.csv"
    if not path.exists():
        return []
    return _records(pd.read_csv(path))


def _load_breakdown() -> list[dict]:
    path = DIAG / "match_quality_breakdown.csv"
    if not path.exists():
        return []
    return _records(pd.read_csv(path))


def _load_model_metrics() -> tuple[dict, str | None]:
    latest = MODELS / "latest.pkl"
    if not latest.exists():
        return {}, None
    with open(latest, "rb") as f:
        bundle = pickle.load(f)
    return bundle.get("metrics", {}), bundle.get("trained_at")


def export() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    rankings = pd.read_parquet(PROC / "value_rankings.parquet")
    current_season = int(rankings["season"].max()) if not rankings.empty else None
    metrics, trained_at = _load_model_metrics()

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_season": current_season,
        "model_trained_at": trained_at,
        "model_metrics": metrics,
        "row_counts": {
            "total_current": int(len(rankings)),
            "eligible_for_ranking": int(
                (
                    ~rankings["low_sample"].fillna(False)
                    & rankings["value_score"].notna()
                    & rankings["contract_match_quality"].isin(["exact", "fuzzy_high"])
                ).sum()
            ),
            "low_sample": int(rankings["low_sample"].fillna(False).sum()),
            "no_match": int((rankings["contract_match_quality"] == "no_match").sum()),
        },
        "players": _records(rankings, PLAYER_COLS),
        "run_history": _load_history(),
        "top_movers": _load_movers(),
        "excluded_players": _load_excluded(),
        "match_quality_breakdown": _load_breakdown(),
    }

    out = DOCS / "data.json"
    with open(out, "w") as f:
        # allow_nan=False catches any sneaky NaN that bypassed _clean — the
        # browser can't parse them, so fail the build loudly instead.
        json.dump(data, f, indent=2, allow_nan=False)
    log.info("dashboard data -> %s (%d players)", out, len(data["players"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export()
