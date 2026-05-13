# nfl-moneyball

A roster-value model for NFL skill-position players that compares on-field production to contract cost, with disciplined handling of missing contract data. Retrains itself every day via GitHub Actions and publishes results to a public Streamlit dashboard.

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

The diagnostic file (`data/diagnostic/excluded_players.csv`) is committed on every run so any excluded player can be traced.

## Autonomous daily retraining

A GitHub Actions workflow (`.github/workflows/daily_train.yml`) runs every day at 11:00 UTC. Each run:

1. Pulls fresh stats and contracts
2. Refits the per-position model
3. Rewrites the value rankings, diagnostic file, and summary report
4. Commits the new artifacts back to `main`

Streamlit Community Cloud auto-redeploys whenever `main` updates, so the dashboard reflects the latest training run within a few minutes.

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

streamlit run app/streamlit_app.py
```

## Repo layout

```
src/pipeline/    data fetchers and merger
src/model/       feature builders, training, scoring
src/reporting/   diagnostic and summary writers
app/             Streamlit dashboard
data/processed/  merged player table (committed)
data/diagnostic/ excluded/flagged players for audit (committed)
models/          fitted model artifacts (committed)
reports/         summary.json for the dashboard (committed)
```

## Scope

In scope: QB, RB, WR, TE roster-value rankings on full or in-progress seasons.

Out of scope (for now): offensive line, defense, special teams (public production metrics are weak without PFF), draft-prospect pricing (future work), historical backtests.

## License

MIT
