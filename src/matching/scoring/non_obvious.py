"""Dimension 3: Non-Obvious Connection Score (20% weight).

Primary signal (80%): shared problem domain tags from the domain model.
Secondary signal (20%): semantic embedding similarity.
"""

from __future__ import annotations

import logging

from src.matching.domain_model import non_obvious_tag_score
from src.matching.embeddings import cosine_similarity, get_embedding
from src.matching.models import AttendeeProfile

logger = logging.getLogger(__name__)

TAG_WEIGHT = 0.80
EMBED_WEIGHT = 0.20


def problem_domain_text(profile: AttendeeProfile) -> str:
    """Construct the problem-domain statement for embedding."""
    parts = []
    if profile.company_description:
        parts.append(profile.company_description)
    if profile.needs_vector:
        parts.append(f"Key challenges: {profile.needs_vector.primary_need}.")
    if profile.provides_vector:
        parts.append(f"Building toward: {profile.provides_vector.primary_capability}.")
    if not parts:
        parts.append(f"{profile.title} at {profile.company}. {profile.stated_goal}")
    return " ".join(parts)


def score(a: AttendeeProfile, b: AttendeeProfile) -> float:
    """Compute non-obvious connection score for a pair.  [0.0, 1.0]."""
    positions_a = a.value_chain_positions or []
    positions_b = b.value_chain_positions or []

    tag_score, shared_tags = non_obvious_tag_score(
        positions_a, positions_b, a.sector, b.sector,
    )

    text_a = problem_domain_text(a)
    text_b = problem_domain_text(b)
    try:
        emb_a = get_embedding(text_a)
        emb_b = get_embedding(text_b)
        embed_sim = cosine_similarity(emb_a, emb_b)
    except RuntimeError:
        embed_sim = 0.0

    combined = TAG_WEIGHT * tag_score + EMBED_WEIGHT * embed_sim
    result = min(max(combined, 0.0), 1.0)

    logger.debug(
        "Non-obvious %s<->%s: tags=%.3f (shared=%s) embed=%.3f -> %.3f",
        a.name, b.name, tag_score,
        ",".join(sorted(shared_tags)) if shared_tags else "none",
        embed_sim, result,
    )
    return result
