"""Dimension 1: Complementarity Score (50% weight).

Three sub-scores:
  - NEEDS <-> PROVIDES alignment (semantic embedding similarity)    50%
  - Value chain adjacency (deterministic lookup)                    30%
  - Bidirectional multiplier                                        20%
"""

from __future__ import annotations

import logging

from src.matching.config import settings
from src.matching.domain_model import best_chain_score
from src.matching.embeddings import cosine_similarity, get_embedding
from src.matching.models import AttendeeProfile

logger = logging.getLogger(__name__)


def _needs_provides_text(profile: AttendeeProfile, vector_type: str) -> str:
    if vector_type == "needs" and profile.needs_vector:
        nv = profile.needs_vector
        return (
            f"{nv.primary_need}. {nv.need_description} "
            f"Looking for: {nv.target_counterparty_type}. "
            f"Constraints: {', '.join(nv.constraints) if nv.constraints else 'none'}."
        )
    if vector_type == "provides" and profile.provides_vector:
        pv = profile.provides_vector
        return (
            f"{pv.primary_capability}. {pv.capability_description} "
            f"Evidence: {', '.join(pv.evidence) if pv.evidence else 'none'}. "
            f"Reach: {', '.join(pv.geographic_reach) if pv.geographic_reach else 'global'}."
        )
    return f"{profile.title} at {profile.company}. {profile.stated_goal}"


def score(a: AttendeeProfile, b: AttendeeProfile) -> float:
    """Compute complementarity score for A's perspective on B.  [0.0, 1.0]."""
    weights = settings.complementarity_weights

    a_needs_text = _needs_provides_text(a, "needs")
    b_provides_text = _needs_provides_text(b, "provides")
    a_provides_text = _needs_provides_text(a, "provides")
    b_needs_text = _needs_provides_text(b, "needs")

    a_needs_emb = get_embedding(a_needs_text)
    b_provides_emb = get_embedding(b_provides_text)
    a_provides_emb = get_embedding(a_provides_text)
    b_needs_emb = get_embedding(b_needs_text)

    ab_alignment = cosine_similarity(a_needs_emb, b_provides_emb)
    ba_alignment = cosine_similarity(b_needs_emb, a_provides_emb)
    avg_alignment = (ab_alignment + ba_alignment) / 2.0

    positions_a = a.value_chain_positions or []
    positions_b = b.value_chain_positions or []
    chain_score = best_chain_score(positions_a, positions_b)

    bidir_raw = 0.0
    if (
        ab_alignment > settings.bidirectional_threshold
        and ba_alignment > settings.bidirectional_threshold
    ):
        bidir_raw = min(avg_alignment * settings.bidirectional_boost, 1.0)
    else:
        bidir_raw = avg_alignment

    composite = (
        weights.needs_provides_alignment * avg_alignment
        + weights.value_chain_adjacency * chain_score
        + weights.bidirectional_multiplier * bidir_raw
    )

    result = min(max(composite, 0.0), 1.0)
    logger.debug(
        "Complementarity %s->%s: align=%.3f chain=%.3f bidir=%.3f -> %.3f",
        a.name, b.name, avg_alignment, chain_score, bidir_raw, result,
    )
    return result
