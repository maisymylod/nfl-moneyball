"""Append a row to run_history.parquet and snapshot today's rankings.

The Streamlit Analytics tab reads these files to plot model performance,
data-quality trends, and per-player value movement over time.
"""
from __future__ import annotations

import logging
import pickle
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

PROC = Path("data/processed")
SNAPSHOTS = PROC / "snapshots"
MODELS = Path("models")
HISTORY_PATH = PROC / "run_history.parquet"
SNAPSHOT_LIMIT = 90  # keep ~3 months of daily snapshots

log = logging.getLogger(__name__)


def _build_row() -> dict:
    rankings = pd.read_parquet(PROC / "value_rankings.parquet")
    current_season = int(rankings["season"].max()) if not rankings.empty else None

    eligible = rankings[
        ~rankings["low_sample"].fillna(False)
        & rankings["value_score"].notna()
        & rankings["contract_match_quality"].isin(["exact", "fuzzy_high"])
    ]

    row = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "run_date": date.today().isoformat(),
        "current_season": current_season,
        "n_current": int(len(rankings)),
        "n_eligible": int(len(eligible)),
        "n_low_sample": int(rankings["low_sample"].fillna(False).sum()),
        "n_no_match": int((rankings["contract_match_quality"] == "no_match").sum()),
        "n_with_contract": int(rankings["apy"].notna().sum()),
    }

    for pos in ("QB", "RB", "WR", "TE"):
        sub = rankings[rankings["position"] == pos]
        row[f"{pos.lower()}_n_current"] = int(len(sub))
        row[f"{pos.lower()}_n_eligible"] = int(
            (
                ~sub["low_sample"].fillna(False)
                & sub["value_score"].notna()
                & sub["contract_match_quality"].isin(["exact", "fuzzy_high"])
            ).sum()
        )

    latest = MODELS / "latest.pkl"
    if latest.exists():
        with open(latest, "rb") as f:
            bundle = pickle.load(f)
        for pos, stats in bundle.get("metrics", {}).items():
            row[f"{pos.lower()}_r2"] = float(stats.get("r2_train", float("nan")))
            row[f"{pos.lower()}_n_train"] = int(stats.get("n_train", 0))
        row["model_trained_at"] = bundle.get("trained_at")

    return row


def _append_history(row: dict) -> pd.DataFrame:
    new = pd.DataFrame([row])
    if HISTORY_PATH.exists():
        existing = pd.read_parquet(HISTORY_PATH)
        # If we already wrote a row today, replace it rather than duplicate.
        existing = existing[existing["run_date"] != row["run_date"]]
        history = pd.concat([existing, new], ignore_index=True)
    else:
        history = new
    history = history.sort_values("run_at").reset_index(drop=True)
    history.to_parquet(HISTORY_PATH, index=False)
    return history


def _snapshot_today() -> Path:
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    rankings = pd.read_parquet(PROC / "value_rankings.parquet")
    cols = [
        c
        for c in (
            "player_id",
            "player_name",
            "position",
            "team",
            "offense_snaps",
            "production",
            "expected_production",
            "production_residual",
            "apy_millions",
            "value_score",
            "confidence",
            "contract_match_quality",
        )
        if c in rankings.columns
    ]
    snap = rankings[cols].copy()
    snap["snapshot_date"] = date.today().isoformat()
    out = SNAPSHOTS / f"{date.today().isoformat()}.parquet"
    snap.to_parquet(out, index=False)

    snaps = sorted(SNAPSHOTS.glob("*.parquet"))
    for old in snaps[:-SNAPSHOT_LIMIT]:
        old.unlink()
    return out


def write_history() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    row = _build_row()
    history = _append_history(row)
    snap_path = _snapshot_today()
    log.info("history rows: %d -> %s", len(history), HISTORY_PATH)
    log.info("snapshot -> %s", snap_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    write_history()
