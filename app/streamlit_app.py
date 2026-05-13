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
        st.dataframe(view, hide_index=True, use_container_width=True)


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
        st.dataframe(excluded, hide_index=True, use_container_width=True)

    breakdown = load_breakdown()
    if not breakdown.empty:
        st.write("**Contract match quality, by position**")
        st.dataframe(breakdown, hide_index=True, use_container_width=True)


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

    tabs = st.tabs(["Bargains", "Overpaid", "Player lookup", "Diagnostic"])
    with tabs[0]:
        tab_bargains(filtered)
    with tabs[1]:
        tab_overpaid(filtered)
    with tabs[2]:
        tab_lookup(filtered)
    with tabs[3]:
        tab_diagnostic()


if __name__ == "__main__":
    main()
