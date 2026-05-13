"""Verify the defensibility rules from the LinkedIn post are upheld."""
from __future__ import annotations

import pandas as pd

from src.pipeline.merge import (
    _attach_contracts,
    _match_quality,
    _add_flags,
    normalize_name,
)


def test_normalize_name_strips_suffixes_and_accents():
    assert normalize_name("Patrick Mahomes II") == "patrick mahomes"
    assert normalize_name("D'Andre Swift Jr.") == "dandre swift"
    assert normalize_name("Amon-Ra St. Brown") == "amonra st brown"
    assert normalize_name(None) == ""


def test_match_quality_buckets():
    assert _match_quality(100) == "exact"
    assert _match_quality(95) == "fuzzy_high"
    assert _match_quality(85) == "fuzzy_low"
    assert _match_quality(70) == "no_match"


def test_low_sample_flag():
    df = pd.DataFrame({"offense_snaps": [50, 99, 100, 800, None]})
    out = _add_flags(df.copy())
    assert out["low_sample"].tolist() == [True, True, False, False, True]


def test_missing_contracts_stay_nan(tmp_path, monkeypatch):
    """The core defensibility check: a player with no contract match must have
    NaN apy, not 0."""
    from src.pipeline import merge as merge_mod

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    contracts = pd.DataFrame(
        {
            "player_name": ["Patrick Mahomes"],
            "position": ["QB"],
            "apy": [45_000_000.0],
            "cap_hit": [None],
            "contract_source": ["overthecap"],
        }
    )
    contracts.to_parquet(raw_dir / "contracts.parquet", index=False)
    monkeypatch.setattr(merge_mod, "RAW", raw_dir)

    df = pd.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "player_name": ["Patrick Mahomes", "Some Backup"],
            "name_norm": ["patrick mahomes", "some backup"],
            "position": ["QB", "QB"],
            "season": [2025, 2025],
        }
    )
    out = merge_mod._attach_contracts(df, season=2025)
    mahomes = out[out["player_name"] == "Patrick Mahomes"].iloc[0]
    backup = out[out["player_name"] == "Some Backup"].iloc[0]

    assert mahomes["apy"] == 45_000_000.0
    assert mahomes["contract_match_quality"] in {"exact", "fuzzy_high"}
    assert pd.isna(backup["apy"]), "unmatched player must keep NaN apy"
    assert backup["contract_match_quality"] == "no_match"
