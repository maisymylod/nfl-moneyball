"""Fit per-position expected-production regressors and persist them."""
from __future__ import annotations

import datetime as dt
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.model.features import FEATURE_COLS, SKILL_POSITIONS, TARGET, build_features

PROC = Path("data/processed")
MODELS = Path("models")
HISTORY = MODELS / "history"
HISTORY_LIMIT = 30

log = logging.getLogger(__name__)


def _fit(df_pos: pd.DataFrame) -> tuple[Pipeline, dict]:
    train = df_pos.dropna(subset=["prior_production", "prior_snaps"]).copy()
    X = train[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").values
    y = pd.to_numeric(train[TARGET], errors="coerce").fillna(0).values
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "gbm",
                GradientBoostingRegressor(
                    n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    r2 = float(pipe.score(X, y))
    return pipe, {"n_train": int(len(train)), "r2_train": r2}


def train() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    HISTORY.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(PROC / "players.parquet")
    df = build_features(raw)

    models: dict[str, Pipeline] = {}
    metrics: dict[str, dict] = {}
    for pos in SKILL_POSITIONS:
        df_pos = df[df["position"] == pos]
        if df_pos.empty or df_pos["prior_production"].notna().sum() < 10:
            log.warning("skipping %s: not enough training rows", pos)
            continue
        m, stats = _fit(df_pos)
        models[pos] = m
        metrics[pos] = stats
        log.info("%s: n=%d, r2=%.3f", pos, stats["n_train"], stats["r2_train"])

    today = dt.date.today().isoformat()
    bundle = {
        "models": models,
        "metrics": metrics,
        "trained_at": today,
        "feature_cols": FEATURE_COLS,
    }
    with open(MODELS / "latest.pkl", "wb") as f:
        pickle.dump(bundle, f)
    with open(HISTORY / f"{today}.pkl", "wb") as f:
        pickle.dump(bundle, f)

    history = sorted(HISTORY.glob("*.pkl"))
    for old in history[:-HISTORY_LIMIT]:
        old.unlink()

    log.info("saved models -> %s", MODELS / "latest.pkl")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
