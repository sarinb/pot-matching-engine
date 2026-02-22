"""Dimension 2: Transaction-Readiness Score (30% weight).

Fully deterministic — evaluates four sub-signals from structured profile
fields using the domain model.  No LLM calls.
"""

from __future__ import annotations

import logging

from src.matching.domain_model import (
    best_transaction_type,
    infer_mandate_score,
    infer_maturity_score,
    stage_compatibility,
)
from src.matching.models import AttendeeProfile

logger = logging.getLogger(__name__)

_WEIGHTS: dict[str, dict[str, float]] = {
    "investment":               {"mandate": 0.30, "fit": 0.30, "maturity": 0.25, "alignment": 0.15},
    "fundraising":              {"mandate": 0.30, "fit": 0.30, "maturity": 0.25, "alignment": 0.15},
    "co_investment":            {"mandate": 0.30, "fit": 0.30, "maturity": 0.25, "alignment": 0.15},
    "partnership":              {"mandate": 0.20, "fit": 0.25, "maturity": 0.30, "alignment": 0.25},
    "gtm_partnership":          {"mandate": 0.20, "fit": 0.25, "maturity": 0.30, "alignment": 0.25},
    "tech_partnership":         {"mandate": 0.20, "fit": 0.25, "maturity": 0.30, "alignment": 0.25},
    "integration":              {"mandate": 0.20, "fit": 0.25, "maturity": 0.30, "alignment": 0.25},
    "technical_collaboration":  {"mandate": 0.20, "fit": 0.25, "maturity": 0.30, "alignment": 0.25},
    "policy_dialogue":          {"mandate": 0.30, "fit": 0.20, "maturity": 0.25, "alignment": 0.25},
    "sandbox_candidacy":        {"mandate": 0.30, "fit": 0.20, "maturity": 0.25, "alignment": 0.25},
    "sandbox_candidates":       {"mandate": 0.30, "fit": 0.20, "maturity": 0.25, "alignment": 0.25},
    "sandbox_evaluation":       {"mandate": 0.30, "fit": 0.20, "maturity": 0.25, "alignment": 0.25},
    "sandbox_participation":    {"mandate": 0.30, "fit": 0.20, "maturity": 0.25, "alignment": 0.25},
    "policy_coordination":      {"mandate": 0.30, "fit": 0.20, "maturity": 0.25, "alignment": 0.25},
    "media_exposure":           {"mandate": 0.20, "fit": 0.25, "maturity": 0.30, "alignment": 0.25},
    "content_collaboration":    {"mandate": 0.20, "fit": 0.25, "maturity": 0.30, "alignment": 0.25},
    "investment_pitch":         {"mandate": 0.30, "fit": 0.30, "maturity": 0.25, "alignment": 0.15},
    "tech_evaluation":          {"mandate": 0.20, "fit": 0.25, "maturity": 0.30, "alignment": 0.25},
}
_DEFAULT_WEIGHTS = {"mandate": 0.25, "fit": 0.25, "maturity": 0.25, "alignment": 0.25}


def _score_mandate(a: AttendeeProfile, b: AttendeeProfile) -> float:
    a_authority = infer_mandate_score(a.title)
    b_authority = infer_mandate_score(b.title)
    a_urgency_bonus = 0.0
    if a.needs_vector and a.needs_vector.urgency == "active":
        a_urgency_bonus = 0.1
    elif a.needs_vector and a.needs_vector.urgency == "exploring":
        a_urgency_bonus = 0.05
    return min((a_authority + b_authority) / 2.0 + a_urgency_bonus, 1.0)


def _score_fit(a: AttendeeProfile, b: AttendeeProfile) -> float:
    fit = 0.5
    if a.needs_vector and b.provides_vector:
        a_target = a.needs_vector.target_counterparty_type.lower()
        b_cap = b.provides_vector.primary_capability.lower()
        a_words = set(a_target.split())
        b_words = set(b_cap.split())
        overlap = len(a_words & b_words)
        if overlap >= 2:
            fit += 0.3
        elif overlap >= 1:
            fit += 0.15
    if a.sector and b.sector and a.sector == b.sector:
        fit += 0.15
    elif a.sector and b.sector:
        fit += 0.05
    return min(fit, 1.0)


def _score_alignment(a: AttendeeProfile, b: AttendeeProfile) -> float:
    alignment = 0.4
    if a.sector and b.sector and a.sector == b.sector:
        alignment += 0.2
    if a.provides_vector and b.provides_vector:
        a_geo = set(a.provides_vector.geographic_reach)
        b_geo = set(b.provides_vector.geographic_reach)
        geo_overlap = a_geo & b_geo
        if geo_overlap and "global" not in geo_overlap:
            alignment += 0.2
        elif geo_overlap:
            alignment += 0.1
    return min(alignment, 1.0)


def score(
    a: AttendeeProfile, b: AttendeeProfile,
) -> tuple[float, str | None]:
    """Compute transaction-readiness score.  Returns (score, transaction_type)."""
    roles_a = a.roles_at_event or ([a.role_at_event] if a.role_at_event else ["other"])
    roles_b = b.roles_at_event or ([b.role_at_event] if b.role_at_event else ["other"])
    tx_type = best_transaction_type(roles_a, roles_b)

    if tx_type is None:
        logger.debug("No transaction type for %s->%s", a.name, b.name)
        return 0.0, None

    mandate = _score_mandate(a, b)
    fit = _score_fit(a, b)
    maturity_a = infer_maturity_score(a.key_facts, a.stage)
    maturity_b = infer_maturity_score(b.key_facts, b.stage)
    maturity = (maturity_a + maturity_b) / 2.0
    alignment = _score_alignment(a, b)

    weights = _WEIGHTS.get(tx_type, _DEFAULT_WEIGHTS)
    rule_score = (
        weights["mandate"] * mandate
        + weights["fit"] * fit
        + weights["maturity"] * maturity
        + weights["alignment"] * alignment
    )

    stage_score = stage_compatibility(tx_type, a.stage, b.stage)
    blended = 0.7 * rule_score + 0.3 * stage_score
    result = min(max(blended, 0.0), 1.0)

    logger.debug(
        "Transaction-readiness %s->%s (%s): mandate=%.2f fit=%.2f "
        "maturity=%.2f align=%.2f stage=%.2f -> %.3f",
        a.name, b.name, tx_type, mandate, fit, maturity, alignment,
        stage_score, result,
    )
    return result, tx_type
