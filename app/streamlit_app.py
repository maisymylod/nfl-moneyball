"""Public dashboard for nfl-moneyball.

Reads parquet + json artifacts that the daily training job commits to the repo,
so this app has no data-fetching logic of its own and works on Streamlit
Community Cloud with no secrets.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RANKINGS_PATH = ROOT / "data" / "processed" / "value_rankings.parquet"
SUMMARY_PATH = ROOT / "reports" / "summary.json"
EXCLUDED_PATH = ROOT / "data" / "diagnostic" / "excluded_players.csv"
BREAKDOWN_PATH = ROOT / "data" / "diagnostic" / "match_quality_breakdown.csv"
HISTORY_PATH = ROOT / "data" / "processed" / "run_history.parquet"
SNAPSHOTS_DIR = ROOT / "data" / "processed" / "snapshots"

CONFIDENCE_COLORS = {
    "High": "#0a7f3f",
    "Medium": "#b8860b",
    "Low": "#b03030",
}

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


@st.cache_data(ttl=600)
def load_rankings() -> pd.DataFrame:
    if not RANKINGS_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(RANKINGS_PATH)


@st.cache_data(ttl=600)
def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        return {}
    with open(SUMMARY_PATH) as f:
        return json.load(f)


@st.cache_data(ttl=600)
def load_excluded() -> pd.DataFrame:
    if not EXCLUDED_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(EXCLUDED_PATH)


@st.cache_data(ttl=600)
def load_breakdown() -> pd.DataFrame:
    if not BREAKDOWN_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(BREAKDOWN_PATH)


@st.cache_data(ttl=600)
def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(HISTORY_PATH)
    df["run_date"] = pd.to_datetime(df["run_date"])
    return df.sort_values("run_date").reset_index(drop=True)


@st.cache_data(ttl=600)
def load_snapshot(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def list_snapshots() -> list[Path]:
    if not SNAPSHOTS_DIR.exists():
        return []
    return sorted(SNAPSHOTS_DIR.glob("*.parquet"))


def confidence_badge(label: str) -> str:
    color = CONFIDENCE_COLORS.get(label, "#888")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.85em;">{label}</span>'


def render_table(df: pd.DataFrame, *, show_badge: bool = True) -> None:
    if df.empty:
        st.info("No rows for the current filters.")
        return
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    view = df[cols].copy()
    for c in view.select_dtypes(include="number").columns:
        view[c] = view[c].astype(float).round(2)
    if show_badge and "confidence" in view.columns:
        view["confidence"] = view["confidence"].apply(confidence_badge)
        st.write(view.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.dataframe(view, hide_index=True, width="stretch")


def header(summary: dict) -> None:
    st.title("nfl-moneyball")
    st.caption("Roster-value model for NFL skill positions. Retrains daily via GitHub Actions.")

    cols = st.columns(4)
    cols[0].metric("Season", summary.get("current_season", "-"))
    counts = summary.get("row_counts", {})
    cols[1].metric("Players (current)", counts.get("total_current", "-"))
    cols[2].metric("Eligible for ranking", counts.get("eligible_for_ranking", "-"))
    cols[3].metric("Trained", summary.get("model_trained_at", "-"))


def filters_panel(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        positions = st.multiselect(
            "Position", ["QB", "RB", "WR", "TE"], default=["QB", "RB", "WR", "TE"]
        )
        min_snaps = st.slider("Minimum offense snaps", 0, 1200, 200, step=50)
        min_confidence = st.selectbox(
            "Minimum confidence", ["Low", "Medium", "High"], index=1
        )
        require_contract = st.checkbox("Only matched contracts", value=True)

    order = {"Low": 0, "Medium": 1, "High": 2}
    threshold = order[min_confidence]

    filtered = df[df["position"].isin(positions)].copy()
    if "offense_snaps" in filtered.columns:
        filtered = filtered[filtered["offense_snaps"].fillna(0) >= min_snaps]
    if "confidence" in filtered.columns:
        filtered = filtered[filtered["confidence"].map(order).fillna(-1) >= threshold]
    if require_contract:
        filtered = filtered[
            filtered["contract_match_quality"].isin(["exact", "fuzzy_high"])
        ]
    return filtered


def tab_bargains(df: pd.DataFrame) -> None:
    st.subheader("Top bargains")
    st.caption(
        "Players whose production exceeds the per-position model's prediction by the most, "
        "scaled by APY. NaN value scores (missing contracts) are excluded by default."
    )
    bargains = (
        df[df["value_score"].notna()]
        .sort_values("value_score", ascending=False)
        .head(50)
    )
    render_table(bargains)


def tab_overpaid(df: pd.DataFrame) -> None:
    st.subheader("Top overpaid")
    st.caption("Players whose production trails the per-position model's prediction the most, scaled by APY.")
    overpaid = (
        df[df["value_score"].notna()]
        .sort_values("value_score", ascending=True)
        .head(50)
    )
    render_table(overpaid)


def tab_lookup(df: pd.DataFrame) -> None:
    st.subheader("Player lookup")
    if df.empty:
        st.info("No data loaded.")
        return
    name = st.selectbox("Player", sorted(df["player_name"].dropna().unique()))
    row = df[df["player_name"] == name]
    render_table(row, show_badge=True)


def tab_analytics() -> None:
    st.subheader("Activity & analytics")
    st.caption(
        "Every daily training run appends a row to `data/processed/run_history.parquet` "
        "and writes a per-player snapshot to `data/processed/snapshots/`. The charts below "
        "are built from those committed artifacts so anyone cloning the repo can reproduce them."
    )

    history = load_history()
    if history.empty:
        st.info(
            "No run history yet. The first daily-train job will seed "
            "`data/processed/run_history.parquet`."
        )
        return

    cols = st.columns(4)
    last = history.iloc[-1]
    cols[0].metric("Runs recorded", len(history))
    cols[1].metric("Most recent run", str(last["run_date"].date()))
    cols[2].metric("Eligible players", int(last["n_eligible"]))
    cols[3].metric("Matched contracts", int(last["n_with_contract"]))

    st.markdown("**Players in the ranked table over time**")
    counts = (
        history.set_index("run_date")[["n_eligible", "n_low_sample", "n_no_match"]]
        .rename(
            columns={
                "n_eligible": "Eligible",
                "n_low_sample": "Low sample",
                "n_no_match": "No contract match",
            }
        )
    )
    st.line_chart(counts)

    r2_cols = [c for c in history.columns if c.endswith("_r2")]
    if r2_cols:
        st.markdown("**Model R² (in-sample) per position over time**")
        r2 = history.set_index("run_date")[r2_cols]
        r2.columns = [c.replace("_r2", "").upper() for c in r2.columns]
        st.line_chart(r2)

    train_cols = [c for c in history.columns if c.endswith("_n_train")]
    if train_cols:
        st.markdown("**Training-set size per position over time**")
        n = history.set_index("run_date")[train_cols]
        n.columns = [c.replace("_n_train", "").upper() for c in n.columns]
        st.line_chart(n)

    snapshots = list_snapshots()
    if len(snapshots) >= 2:
        st.markdown("**Top movers since prior run**")
        cur = load_snapshot(snapshots[-1])
        prev = load_snapshot(snapshots[-2])
        merged = cur.merge(
            prev[["player_id", "value_score", "production_residual"]],
            on="player_id",
            how="inner",
            suffixes=("", "_prev"),
        )
        merged["value_score_delta"] = (
            merged["value_score"] - merged["value_score_prev"]
        )
        movers = merged.dropna(subset=["value_score_delta"]).copy()
        if not movers.empty:
            top_up = movers.sort_values("value_score_delta", ascending=False).head(10)
            top_down = movers.sort_values("value_score_delta", ascending=True).head(10)
            mover_cols = [
                "player_name",
                "position",
                "team",
                "value_score_prev",
                "value_score",
                "value_score_delta",
                "confidence",
            ]
            up_cols = st.columns(2)
            with up_cols[0]:
                st.write("Risers")
                view = top_up[mover_cols].copy()
                for c in view.select_dtypes(include="number").columns:
                    view[c] = view[c].astype(float).round(2)
                st.dataframe(view, hide_index=True, width="stretch")
            with up_cols[1]:
                st.write("Fallers")
                view = top_down[mover_cols].copy()
                for c in view.select_dtypes(include="number").columns:
                    view[c] = view[c].astype(float).round(2)
                st.dataframe(view, hide_index=True, width="stretch")
        else:
            st.info("No overlapping players between the last two snapshots.")
    else:
        st.info(
            f"Need at least 2 daily snapshots to compute top movers. Currently have {len(snapshots)}."
        )

    st.markdown("**Full run history**")
    display = history.copy()
    display["run_date"] = display["run_date"].dt.strftime("%Y-%m-%d")
    for c in display.select_dtypes(include="float").columns:
        display[c] = display[c].round(3)
    st.dataframe(display, hide_index=True, width="stretch")


def tab_diagnostic() -> None:
    st.subheader("Diagnostic")
    st.caption(
        "Players excluded from the ranked tables are listed here for audit. "
        "Reason codes: low_sample (under 100 snaps), no_contract_match (no public "
        "contract record), fuzzy_match_below_threshold."
    )
    excluded = load_excluded()
    if excluded.empty:
        st.info("No exclusion file yet. Run the pipeline.")
    else:
        st.write(f"**Excluded players:** {len(excluded)}")
        st.dataframe(excluded, hide_index=True, width="stretch")

    breakdown = load_breakdown()
    if not breakdown.empty:
        st.write("**Contract match quality, by position**")
        st.dataframe(breakdown, hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="nfl-moneyball", layout="wide")
    summary = load_summary()
    rankings = load_rankings()

    header(summary)

    if rankings.empty:
        st.warning(
            "No rankings parquet found. Run `python -m src.pipeline.fetch_stats` "
            "through `python -m src.reporting.summary` to populate."
        )
        return

    filtered = filters_panel(rankings)

    tabs = st.tabs(["Bargains", "Overpaid", "Player lookup", "Analytics", "Diagnostic"])
    with tabs[0]:
        tab_bargains(filtered)
    with tabs[1]:
        tab_overpaid(filtered)
    with tabs[2]:
        tab_lookup(filtered)
    with tabs[3]:
        tab_analytics()
    with tabs[4]:
        tab_diagnostic()


if __name__ == "__main__":
    main()
