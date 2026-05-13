"""Write reports/summary.json with headline lists for the dashboard."""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROC = Path("data/processed")
MODELS = Path("models")
REPORTS = Path("reports")

TOP_N = 25
PER_POSITION_N = 5

log = logging.getLogger(__name__)

DISPLAY_COLS = [
    "player_name",
    "position",
    "team",
    "age",
    "offense_snaps",
    "production",
    "expected_production",
    "production_residual",
    "apy_millions",
    "value_score",
    "confidence",
    "contract_match_quality",
]


def _records(df: pd.DataFrame) -> list[dict]:
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    out = df[cols].copy()
    for c in out.select_dtypes(include="number").columns:
        out[c] = out[c].astype(float).round(3)
    return out.where(out.notna(), None).to_dict(orient="records")


def write_summary() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rankings = pd.read_parquet(PROC / "value_rankings.parquet")

    eligible = rankings[
        ~rankings["low_sample"].fillna(False)
        & rankings["value_score"].notna()
        & rankings["contract_match_quality"].isin(["exact", "fuzzy_high"])
    ].copy()

    top_bargains = eligible.sort_values("value_score", ascending=False).head(TOP_N)
    top_overpaid = eligible.sort_values("value_score", ascending=True).head(TOP_N)

    by_position = {}
    for pos in ("QB", "RB", "WR", "TE"):
        sub = eligible[eligible["position"] == pos]
        by_position[pos] = {
            "bargains": _records(sub.sort_values("value_score", ascending=False).head(PER_POSITION_N)),
            "overpaid": _records(sub.sort_values("value_score", ascending=True).head(PER_POSITION_N)),
        }

    model_metrics = {}
    latest = MODELS / "latest.pkl"
    if latest.exists():
        with open(latest, "rb") as f:
            bundle = pickle.load(f)
        model_metrics = bundle.get("metrics", {})
        trained_at = bundle.get("trained_at")
    else:
        trained_at = None

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_trained_at": trained_at,
        "current_season": int(rankings["season"].max()) if not rankings.empty else None,
        "row_counts": {
            "total_current": int(len(rankings)),
            "eligible_for_ranking": int(len(eligible)),
            "low_sample": int(rankings["low_sample"].fillna(False).sum()),
            "no_match": int((rankings["contract_match_quality"] == "no_match").sum()),
        },
        "model_metrics": model_metrics,
        "top_bargains": _records(top_bargains),
        "top_overpaid": _records(top_overpaid),
        "by_position": by_position,
    }

    out = REPORTS / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("summary -> %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    write_summary()
