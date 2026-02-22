"""Explanation generator — structured scores to natural-language rationale.

Generates a 2-3 sentence main explanation PLUS one-liner explanations for
each scoring dimension (complementarity, transaction-readiness, non-obvious).
"""

from __future__ import annotations

import json
import logging

from anthropic import Anthropic

from src.matching.llm import call_llm_json
from src.matching.models import (
    AttendeeProfile,
    DimensionExplanations,
    ScoredPair,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You generate match explanations for a premium Web3 conference matchmaking engine.
You receive scoring data for a match between two attendees and must produce:

1. A 2-3 sentence strategic explanation of why Attendee A should meet Attendee B.
2. A one-liner for the Complementarity dimension (needs/provides fit).
3. A one-liner for the Transaction-Readiness dimension (ability to act).
4. A one-liner for the Non-Obvious dimension (unexpected connection value).

RULES:
- Reference SPECIFIC, verifiable facts (funding amounts, customer names, mandates).
- Explain MUTUAL value — not just shared interest.
- Hedge when data confidence is medium/low ("likely", "suggests").
- Never fabricate facts not present in the profile data.
- No generic phrases like "you both share an interest in..."
- Dimension one-liners should be max 15 words each.
- If a dimension score is 0 or near-zero, say why it's low honestly.

Return ONLY valid JSON:
{
  "explanation": "2-3 sentence main explanation",
  "complementarity": "one-liner for D1",
  "transaction_readiness": "one-liner for D2",
  "non_obvious": "one-liner for D3"
}
"""


def _build_user_message(
    pair: ScoredPair,
    perspective: AttendeeProfile,
    other: AttendeeProfile,
) -> str:
    parts = [
        f"PERSPECTIVE ATTENDEE (who receives this recommendation):",
        f"  Name: {perspective.name}",
        f"  Title: {perspective.title}, {perspective.company}",
    ]
    if perspective.product:
        parts.append(f"  Product: {perspective.product}")
    if perspective.stated_goal:
        parts.append(f"  Goal: {perspective.stated_goal}")
    if perspective.needs_vector:
        parts.append(f"  Needs: {perspective.needs_vector.primary_need}")
    if perspective.key_facts:
        parts.append(f"  Key facts: {'; '.join(perspective.key_facts)}")

    parts.append("")
    parts.append("RECOMMENDED MATCH:")
    parts.append(f"  Name: {other.name}")
    parts.append(f"  Title: {other.title}, {other.company}")
    if other.product:
        parts.append(f"  Product: {other.product}")
    if other.stated_goal:
        parts.append(f"  Goal: {other.stated_goal}")
    if other.provides_vector:
        parts.append(f"  Provides: {other.provides_vector.primary_capability}")
        if other.provides_vector.evidence:
            parts.append(f"  Evidence: {'; '.join(other.provides_vector.evidence)}")
    if other.key_facts:
        parts.append(f"  Key facts: {'; '.join(other.key_facts)}")

    parts.append("")
    parts.append("SCORES:")
    parts.append(f"  Composite: {pair.composite:.2f}")
    parts.append(f"  Complementarity: {pair.scores.complementarity:.2f}")
    parts.append(f"  Transaction-Readiness: {pair.scores.transaction_readiness:.2f}")
    parts.append(f"  Non-Obvious: {pair.scores.non_obvious:.2f}")
    if pair.transaction_type:
        parts.append(f"  Transaction type: {pair.transaction_type}")

    return "\n".join(parts)


def generate_explanation(
    client: Anthropic,
    pair: ScoredPair,
    perspective: AttendeeProfile,
    other: AttendeeProfile,
) -> tuple[str, DimensionExplanations]:
    """Generate main explanation + dimension one-liners.

    Returns (explanation_text, DimensionExplanations).
    """
    user_msg = _build_user_message(pair, perspective, other)
    data = call_llm_json(client, _SYSTEM_PROMPT, user_msg, fast=False)

    explanation = data.get("explanation", "")
    dim_explanations = DimensionExplanations(
        complementarity=data.get("complementarity", ""),
        transaction_readiness=data.get("transaction_readiness", ""),
        non_obvious=data.get("non_obvious", ""),
    )

    if not explanation:
        logger.warning(
            "Empty explanation for %s -> %s", perspective.name, other.name,
        )
        explanation = (
            f"{other.name} ({other.company}) may be relevant to your goals "
            f"at this event."
        )

    return explanation, dim_explanations
