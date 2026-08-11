# nfl-moneyball

**Live dashboard:** [maisymylod.github.io/nfl-moneyball](https://maisymylod.github.io/nfl-moneyball/)

A roster-value model for NFL skill-position players that compares on-field production to contract cost, with disciplined handling of missing contract data. Retrains itself every day via GitHub Actions and publishes results to a static dashboard hosted on GitHub Pages.

## What it does

For every QB, RB, WR, and TE with meaningful playing time:

1. Pulls seasonal stats (via `nfl_data_py`, the Python port of nflfastR)
2. Pulls contract data (APY, cap hit, years remaining) from OverTheCap
3. Fits a per-position model of expected production given age, experience, snaps, and prior-year output
4. Computes a value score: production residual normalized by APY
5. Tags each player with a High / Medium / Low confidence label

## Defensibility

Public NFL contract data is messy. If missing values get imputed as $0, a player who simply isn't matched to a contract record looks artificially cheap and floats to the top of the rankings. This project takes that seriously:

- Missing contract values stay missing (never imputed to $0)
- Every player carries a `contract_match_quality` flag (`exact`, `fuzzy_high`, `fuzzy_low`, `no_match`)
- Low-sample players (under 100 snaps) are excluded from the ranked table and routed to a diagnostic file for audit
- Confidence labels surface uncertainty directly in the dashboard

The diagnostic file (`data/diagnostic/excluded_players.csv`) is rewritten on every run so any excluded player can be traced.

## Autonomous daily retraining

A GitHub Actions workflow (`.github/workflows/daily_train.yml`) runs every day at 11:00 UTC. Each run:

1. Restores the accumulated run history and daily snapshots from the `daily-latest` release
2. Pulls fresh stats and contracts
3. Refits the per-position model
4. Rewrites the value rankings, diagnostic file, and summary report
5. Publishes `model-latest.pkl` and the updated `state.tar.gz` back to the `daily-latest` release
6. Deploys the regenerated `docs/` to GitHub Pages

Every one of those outputs is derived from the code and the upstream data, so
none of them are committed: the run publishes them as release assets and a Pages
deployment instead of pushing ~1MB of regenerated binaries to `main` each day.
That keeps `main` to source only, and keeps the artifacts versioned and
downloadable rather than buried in the history.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.pipeline.fetch_stats
python -m src.pipeline.fetch_contracts
python -m src.pipeline.merge
python -m src.model.train
python -m src.model.score
python -m src.reporting.diagnostic
python -m src.reporting.summary

python -m src.reporting.dashboard_export   # compiles docs/data.json

# Static dashboard: open docs/index.html in a browser, or
python -m http.server 8000 --directory docs
# Optional: the legacy Streamlit dashboard
streamlit run app/streamlit_app.py
```

## Repo layout

```
src/pipeline/    data fetchers and merger
src/model/       feature builders, training, scoring
src/reporting/   diagnostic, summary, history, dashboard_export
app/             legacy Streamlit dashboard (optional, run locally)
docs/            static HTML dashboard served by GitHub Pages
data/processed/  merged player table + run history + snapshots  (generated)
data/diagnostic/ excluded/flagged players for audit             (generated)
models/          fitted model artifacts                         (generated)
reports/         summary.json                                   (generated)
```

The four directories marked *generated* are git-ignored. Run the pipeline below
to produce them, or download `state.tar.gz` and `model-latest.pkl` from the
[`daily-latest`](https://github.com/maisymylod/nfl-moneyball/releases/tag/daily-latest)
release to start from the current accumulated history. The optional Streamlit
Analytics tab needs `run_history.parquet` and `snapshots/`, so it wants the
release copy (or enough local runs to build its own).

## Scope

In scope: QB, RB, WR, TE roster-value rankings on full or in-progress seasons.

Out of scope (for now): offensive line, defense, special teams (public production metrics are weak without PFF), draft-prospect pricing (future work), historical backtests.

## License

MIT
