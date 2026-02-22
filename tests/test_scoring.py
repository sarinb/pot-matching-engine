"""Unit tests for the scoring modules — runs without LLM / API calls."""

from __future__ import annotations

from src.matching.models import (
    AttendeeProfile,
    DimensionScores,
    NeedsVector,
    ProvidesVector,
)
from src.matching.scoring.composite import composite_score


def _make_profile(
    name: str,
    title: str = "CEO",
    company: str = "TestCo",
    product: str | None = None,
    need: str = "partnerships",
    provides: str = "technology",
    roles: list[str] | None = None,
    sector: str | None = None,
) -> AttendeeProfile:
    return AttendeeProfile(
        name=name,
        title=title,
        company=company,
        product=product,
        stated_goal=f"Looking for {need}",
        roles_at_event=roles or ["exploring_partnerships"],
        sector=sector,
        needs_vector=NeedsVector(
            primary_need=need,
            need_description=f"Needs {need}",
            target_counterparty_type="partner",
            urgency="exploring",
        ),
        provides_vector=ProvidesVector(
            primary_capability=provides,
            capability_description=f"Provides {provides}",
        ),
    )


class TestCompositeScore:
    def test_zero_scores(self):
        scores = DimensionScores(
            complementarity=0.0, transaction_readiness=0.0, non_obvious=0.0,
        )
        assert composite_score(scores) == 0.0

    def test_perfect_scores(self):
        scores = DimensionScores(
            complementarity=1.0, transaction_readiness=1.0, non_obvious=1.0,
        )
        s = composite_score(scores)
        assert abs(s - 1.0) < 0.01

    def test_complementarity_dominant(self):
        """Higher complementarity should yield a higher composite."""
        high_comp = DimensionScores(
            complementarity=0.9, transaction_readiness=0.3, non_obvious=0.2,
        )
        high_tr = DimensionScores(
            complementarity=0.2, transaction_readiness=0.9, non_obvious=0.2,
        )
        assert composite_score(high_comp) > composite_score(high_tr)


class TestTransactionReadiness:
    def test_no_roles_returns_zero(self):
        from src.matching.scoring.transaction_readiness import score

        a = _make_profile("A", roles=["other"])
        b = _make_profile("B", roles=["other"])
        s, tx = score(a, b)
        assert s == 0.0
        assert tx is None

    def test_valid_pair(self):
        from src.matching.scoring.transaction_readiness import score

        a = _make_profile("Investor", title="GP", roles=["deploying_capital"])
        b = _make_profile("Founder", title="CEO", roles=["raising_capital"])
        s, tx = score(a, b)
        assert s > 0.0
        assert tx == "investment"
