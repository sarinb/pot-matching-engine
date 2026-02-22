"""Dimension 3: Non-Obvious Connection Score (20% weight).

Embeds each attendee's problem domain using local TF-IDF, then uses
cosine similarity to detect structural parallels across industry
boundaries.  A novelty boost applies when two attendees are in
different sectors but have high embedding similarity.
"""

from __future__ import annotations

import logging

from src.matching.config import settings
from src.matching.embeddings import cosine_similarity, get_embedding
from src.matching.models import AttendeeProfile

logger = logging.getLogger(__name__)


def problem_domain_text(profile: AttendeeProfile) -> str:
    """Construct the problem-domain statement for embedding."""
    parts = []

    if profile.company_description:
        parts.append(profile.company_description)

    if profile.needs_vector:
        parts.append(f"Key challenges: {profile.needs_vector.primary_need}.")

    if profile.provides_vector:
        parts.append(
            f"Building toward: {profile.provides_vector.primary_capability}."
        )

    if not parts:
        parts.append(
            f"{profile.title} at {profile.company}. {profile.stated_goal}"
        )

    return " ".join(parts)


def score(
    a: AttendeeProfile,
    b: AttendeeProfile,
) -> float:
    """Compute the non-obvious connection score for a pair.

    Returns a float in [0.0, 1.0].
    """
    text_a = problem_domain_text(a)
    text_b = problem_domain_text(b)

    emb_a = get_embedding(text_a)
    emb_b = get_embedding(text_b)

    similarity = cosine_similarity(emb_a, emb_b)

    # Novelty boost: different sectors + high similarity → cross-boundary insight
    different_sectors = (
        a.sector is not None
        and b.sector is not None
        and a.sector != b.sector
    )
    if different_sectors and similarity > settings.novelty_similarity_threshold:
        boosted = similarity * settings.novelty_boost
        result = min(boosted, 1.0)
        logger.debug(
            "Non-obvious %s↔%s: sim=%.3f (cross-sector boost → %.3f)",
            a.name, b.name, similarity, result,
        )
        return result

    logger.debug(
        "Non-obvious %s↔%s: sim=%.3f (sectors: %s / %s)",
        a.name, b.name, similarity, a.sector, b.sector,
    )
    return max(min(similarity, 1.0), 0.0)
