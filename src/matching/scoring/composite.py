"""Composite ranker — weighted sum of all dimension scores."""

from __future__ import annotations

from src.matching.config import settings
from src.matching.models import DimensionScores


def composite_score(scores: DimensionScores) -> float:
    w = settings.dimension_weights
    return (
        w.complementarity * scores.complementarity
        + w.transaction_readiness * scores.transaction_readiness
        + w.non_obvious * scores.non_obvious
    )
