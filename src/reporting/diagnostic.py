"""Write supplementary audit artifacts for the run.

The primary excluded-players CSV is written by `src.pipeline.merge`. This module
adds breakdowns useful for inspecting how the pipeline is performing over time.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROC = Path("data/processed")
DIAG = Path("data/diagnostic")

log = logging.getLogger(__name__)


def write_diagnostics() -> None:
    DIAG.mkdir(parents=True, exist_ok=True)
    players = pd.read_parquet(PROC / "players.parquet")
    current_season = int(players["season"].max())
    current = players[players["season"] == current_season].copy()

    breakdown = (
        current.groupby(["position", "contract_match_quality"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["position", "contract_match_quality"])
    )
    breakdown.to_csv(DIAG / "match_quality_breakdown.csv", index=False)
    log.info("match-quality breakdown -> %s", DIAG / "match_quality_breakdown.csv")

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_season": current_season,
        "row_counts": {
            "players_all_seasons": int(len(players)),
            "players_current_season": int(len(current)),
            "matched_exact_or_fuzzy_high": int(
                current["contract_match_quality"].isin(["exact", "fuzzy_high"]).sum()
            ),
            "no_match": int((current["contract_match_quality"] == "no_match").sum()),
            "low_sample": int(current["low_sample"].sum()),
        },
        "contract_sources": current["contract_source"]
        .fillna("none")
        .value_counts()
        .to_dict(),
    }
    with open(DIAG / "run_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    log.info("run audit -> %s", DIAG / "run_audit.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    write_diagnostics()
