"""Top-level orchestrator — ties all components together.

Pipeline:
  1. Load / receive profiles
  2. Extract ALL intelligence per profile in one LLM call (Claude Haiku)
  3. Pre-encode all profile texts with sentence-transformers  (local)
  4. Score all pairs across three dimensions                  (deterministic)
  5. Compute composite scores, rank per attendee
  6. Generate explanations for Top-K only                     (Claude Sonnet, optional)
  7. Return results
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from anthropic import Anthropic

from src.matching import embeddings
from src.matching.config import settings
from src.matching.domain_model import best_transaction_type
from src.matching.explanation.generator import generate_explanation
from src.matching.extraction.intent_extractor import extract_all
from src.matching.models import (
    AttendeeProfile,
    DimensionExplanations,
    DimensionScores,
    MatchBriefing,
    MatchResult,
    ScoredPair,
)
from src.matching.scoring import complementarity, non_obvious, transaction_readiness
from src.matching.scoring.composite import composite_score

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_sample_profiles() -> list[AttendeeProfile]:
    path = DATA_DIR / "sample_profiles.json"
    with open(path) as f:
        raw = json.load(f)
    return [AttendeeProfile(**p) for p in raw]


def load_minimal_profiles() -> list[AttendeeProfile]:
    path = DATA_DIR / "sample_profiles_minimal.json"
    with open(path) as f:
        raw = json.load(f)
    return [AttendeeProfile(**p) for p in raw]


def load_profiles_from_json(data: list[dict]) -> list[AttendeeProfile]:
    return [AttendeeProfile(**p) for p in data]


def _collect_embedding_corpus(profiles: list[AttendeeProfile]) -> list[str]:
    texts = []
    for p in profiles:
        texts.append(complementarity._needs_provides_text(p, "needs"))
        texts.append(complementarity._needs_provides_text(p, "provides"))
        texts.append(non_obvious.problem_domain_text(p))
    return texts


def _get_roles(p: AttendeeProfile) -> list[str]:
    if p.roles_at_event:
        return list(p.roles_at_event)
    if p.role_at_event:
        return [p.role_at_event]
    return ["other"]


def run(
    profiles: list[AttendeeProfile],
    client: Anthropic | None = None,
    top_k: int | None = None,
    skip_explanations: bool = False,
    progress_callback: Callable[[str, float], None] | None = None,
) -> list[MatchBriefing]:
    if client is None:
        client = Anthropic(api_key=settings.anthropic_api_key)

    k = top_k or settings.top_k
    n = len(profiles)

    def _progress(label: str, frac: float) -> None:
        if progress_callback:
            progress_callback(label, frac)

    # Step 1: Consolidated extraction
    _progress("Extracting profile intelligence...", 0.0)
    enriched: list[AttendeeProfile] = []
    for i, profile in enumerate(profiles):
        enriched.append(extract_all(client, profile))
        _progress("Extracting profile intelligence...", (i + 1) / n * 0.25)

    # Step 2: Pre-encode embeddings
    _progress("Building embedding model...", 0.25)
    corpus = _collect_embedding_corpus(enriched)
    embeddings.fit(corpus)
    _progress("Building embedding model...", 0.30)

    # Step 3: Score all pairs
    _progress("Scoring pairs...", 0.30)
    profile_map = {p.id: p for p in enriched}
    pair_scores: dict[tuple[str, str], ScoredPair] = {}

    scoreable_pairs = []
    for a in enriched:
        for b in enriched:
            if a.id == b.id:
                continue
            tx_type = best_transaction_type(_get_roles(a), _get_roles(b))
            scoreable_pairs.append((a, b, tx_type))

    total_pairs = len(scoreable_pairs)
    for idx, (a, b, tx_type) in enumerate(scoreable_pairs):
        comp = complementarity.score(a, b)
        no_score = non_obvious.score(a, b)

        if tx_type is None:
            tr_score = 0.0
        else:
            tr_score, _ = transaction_readiness.score(a, b)

        dim_scores = DimensionScores(
            complementarity=comp,
            transaction_readiness=tr_score,
            non_obvious=no_score,
        )
        comp_score = composite_score(dim_scores)

        pair_scores[(a.id, b.id)] = ScoredPair(
            attendee_a_id=a.id,
            attendee_b_id=b.id,
            scores=dim_scores,
            composite=comp_score,
            transaction_type=tx_type,
        )
        _progress("Scoring pairs...", 0.30 + ((idx + 1) / total_pairs) * 0.4)

    # Step 4: Rank and generate explanations
    _progress("Ranking matches...", 0.7)
    briefings: list[MatchBriefing] = []
    attendee_count = 0

    for attendee in enriched:
        candidates = [
            sp for (a_id, _), sp in pair_scores.items()
            if a_id == attendee.id
        ]
        candidates.sort(key=lambda sp: sp.composite, reverse=True)
        top = candidates[:k]

        matches: list[MatchResult] = []
        for sp in top:
            if skip_explanations:
                explanation = "(explanations skipped)"
                dim_explanations = DimensionExplanations()
            else:
                other = profile_map[sp.attendee_b_id]
                explanation, dim_explanations = generate_explanation(
                    client, sp, attendee, other,
                )
            matches.append(MatchResult(
                pair=sp,
                explanation=explanation,
                dimension_explanations=dim_explanations,
            ))

        briefings.append(MatchBriefing(
            attendee_id=attendee.id,
            attendee_name=attendee.name,
            matches=matches,
        ))
        attendee_count += 1
        frac = 0.7 + (attendee_count / n) * 0.3 if not skip_explanations else 0.95
        _progress(
            "Generating explanations..." if not skip_explanations else "Ranking...",
            frac,
        )

    _progress("Complete", 1.0)
    logger.info(
        "Pipeline complete: %d profiles, %d pairs scored, %d briefings "
        "(explanations=%s)",
        n, len(pair_scores), len(briefings),
        "off" if skip_explanations else "on",
    )
    return briefings
