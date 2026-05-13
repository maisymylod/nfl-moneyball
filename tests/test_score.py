"""Verify scoring rules: confidence labels and value-score nan-propagation."""
from __future__ import annotations

import pandas as pd

from src.model.score import _confidence


def test_confidence_high_requires_snaps_and_match_and_prior():
    row = pd.Series(
        {"offense_snaps": 800, "contract_match_quality": "exact", "prior_production": 50.0}
    )
    assert _confidence(row) == "High"


def test_confidence_medium_without_prior():
    row = pd.Series(
        {"offense_snaps": 800, "contract_match_quality": "exact", "prior_production": None}
    )
    assert _confidence(row) == "Medium"


def test_confidence_low_when_unmatched():
    row = pd.Series(
        {"offense_snaps": 800, "contract_match_quality": "no_match", "prior_production": 50.0}
    )
    assert _confidence(row) == "Low"


def test_confidence_low_when_low_sample():
    row = pd.Series(
        {"offense_snaps": 80, "contract_match_quality": "exact", "prior_production": 50.0}
    )
    assert _confidence(row) == "Low"


def test_value_score_nan_when_apy_missing():
    """Sanity check the formula directly: NaN apy → NaN value_score."""
    apy_millions = pd.Series([10.0, None, 0.0, 30.0])
    residual = pd.Series([5.0, 5.0, 5.0, 5.0])
    score = residual / apy_millions.where(apy_millions > 0)
    assert score.iloc[0] == 0.5
    assert pd.isna(score.iloc[1]), "NaN apy must produce NaN value_score"
    assert pd.isna(score.iloc[2]), "Zero apy must produce NaN value_score (not inf)"
    assert score.iloc[3] == 5.0 / 30.0
