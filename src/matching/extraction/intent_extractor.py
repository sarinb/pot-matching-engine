"""Consolidated LLM extraction — one call per profile.

Extracts NEEDS vector, PROVIDES vector, roles at event, and value chain
positions in a single Claude Haiku call.  Results cached by profile hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from anthropic import Anthropic

from src.matching.domain_model import (
    VALUE_CHAINS,
    infer_audience,
    infer_capability,
    infer_geographic_reach,
)
from src.matching.llm import call_llm_json
from src.matching.models import (
    AttendeeProfile,
    NeedsVector,
    ProvidesVector,
    RoleAtEvent,
)

logger = logging.getLogger(__name__)

_cache: dict[str, dict[str, Any]] = {}

_SYSTEM_PROMPT = """\
You are an expert analyst for a Web3/blockchain conference matchmaking engine.
Given an attendee profile, extract structured intelligence in JSON.

Return a single JSON object with these keys:

{
  "needs": {
    "primary_need": "concise label",
    "need_description": "1-2 sentence detail",
    "target_counterparty_type": "who they need to meet",
    "urgency": "active|exploring|passive",
    "constraints": ["specific requirement 1", ...]
  },
  "provides": {
    "primary_capability": "concise label",
    "capability_description": "1-2 sentence detail",
    "evidence": ["verifiable data point 1", ...],
    "geographic_reach": ["region1", ...],
    "audience_access": ["audience type 1", ...]
  },
  "roles_at_event": ["deploying_capital", "raising_capital", "exploring_partnerships",
                      "seeking_technology", "regulatory_policy", "media_content"],
  "value_chain_positions": [["chain_name", "position"], ...]
}

RULES:
- roles_at_event: pick 1-3 from the allowed values based on the profile. Multiple roles are expected.
- value_chain_positions: map to positions in these chains:
  tokenized_securities: issuance, custody, settlement, distribution, compliance_audit
  defi_infrastructure: protocol, scaling_l1_l2, interoperability, wallet_access, compliance
  institutional_adoption: regulatory_framework, sandbox, pilot, production, scale
  capital_markets: fund_formation, deal_sourcing, due_diligence, portfolio_management, exit
- provides: infer from context (title, company, product). Do NOT copy stated_goal into provides.
- urgency: "active" if language shows immediacy/mandates, "exploring" if open/interested, "passive" if vague.
- Return ONLY valid JSON, no markdown fences.
"""


def _profile_hash(profile: AttendeeProfile) -> str:
    key = f"{profile.name}|{profile.title}|{profile.company}|{profile.product}|{profile.stated_goal}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _build_user_message(profile: AttendeeProfile) -> str:
    parts = [
        f"Name: {profile.name}",
        f"Title: {profile.title}",
        f"Company: {profile.company}",
    ]
    if profile.product:
        parts.append(f"Product/offering: {profile.product}")
    if profile.company_description and profile.company_description != profile.product:
        parts.append(f"Company description: {profile.company_description}")
    if profile.stated_goal:
        parts.append(f"Stated goal: {profile.stated_goal}")
    if profile.sector:
        parts.append(f"Sector: {profile.sector}")
    if profile.stage:
        parts.append(f"Stage: {profile.stage}")
    if profile.funding_raised:
        parts.append(f"Funding raised: {profile.funding_raised}")
    if profile.key_facts:
        parts.append(f"Key facts: {'; '.join(profile.key_facts)}")
    if profile.roles_at_event:
        parts.append(f"Self-reported roles: {', '.join(profile.roles_at_event)}")
    return "\n".join(parts)


def _fallback_provides(profile: AttendeeProfile) -> ProvidesVector:
    cap = infer_capability(profile.title, profile.company_description)
    audience = infer_audience(profile.title, profile.company_description)
    geo = infer_geographic_reach(profile.company_description)
    return ProvidesVector(
        primary_capability=cap,
        capability_description=profile.company_description or f"{profile.title} at {profile.company}",
        evidence=profile.key_facts[:3] if profile.key_facts else [],
        geographic_reach=geo,
        audience_access=audience,
    )


def _parse_needs(data: dict) -> NeedsVector:
    return NeedsVector(
        primary_need=data.get("primary_need", ""),
        need_description=data.get("need_description", ""),
        target_counterparty_type=data.get("target_counterparty_type", ""),
        urgency=data.get("urgency", "exploring"),
        constraints=data.get("constraints", []),
    )


def _parse_provides(data: dict) -> ProvidesVector:
    return ProvidesVector(
        primary_capability=data.get("primary_capability", ""),
        capability_description=data.get("capability_description", ""),
        evidence=data.get("evidence", []),
        geographic_reach=data.get("geographic_reach", []),
        audience_access=data.get("audience_access", []),
    )


_VALID_ROLES: set[str] = {
    "deploying_capital", "raising_capital", "exploring_partnerships",
    "seeking_technology", "regulatory_policy", "media_content", "other",
}

_VALID_POSITIONS: dict[str, set[str]] = {
    chain: set(positions) for chain, positions in VALUE_CHAINS.items()
}


def extract_all(client: Anthropic, profile: AttendeeProfile) -> AttendeeProfile:
    """Run consolidated extraction, returning an enriched copy of the profile."""
    h = _profile_hash(profile)
    if h in _cache:
        cached = _cache[h]
        return _apply_extraction(profile, cached)

    user_msg = _build_user_message(profile)
    result = call_llm_json(client, _SYSTEM_PROMPT, user_msg, fast=True)

    if not result:
        logger.warning("Empty LLM response for %s — using fallbacks", profile.name)
        result = {}

    _cache[h] = result
    return _apply_extraction(profile, result)


def _apply_extraction(
    profile: AttendeeProfile, data: dict,
) -> AttendeeProfile:
    # NEEDS
    needs_data = data.get("needs", {})
    if needs_data and needs_data.get("primary_need"):
        needs = _parse_needs(needs_data)
    else:
        goal_text = profile.stated_goal or profile.title
        needs = NeedsVector(
            primary_need=goal_text,
            need_description=goal_text,
            target_counterparty_type="",
            urgency="exploring",
        )

    # PROVIDES
    provides_data = data.get("provides", {})
    if provides_data and provides_data.get("primary_capability"):
        provides = _parse_provides(provides_data)
    else:
        provides = _fallback_provides(profile)

    # ROLES
    raw_roles = data.get("roles_at_event", [])
    roles = [r for r in raw_roles if r in _VALID_ROLES]
    if not roles:
        roles = list(profile.roles_at_event or [])
    if not roles:
        roles = ["other"]

    # VALUE CHAIN POSITIONS
    raw_positions = data.get("value_chain_positions", [])
    positions: list[tuple[str, str]] = []
    for item in raw_positions:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            chain, pos = item[0], item[1]
            if chain in _VALID_POSITIONS and pos in _VALID_POSITIONS[chain]:
                positions.append((chain, pos))

    updated = profile.model_copy(
        update={
            "needs_vector": needs,
            "provides_vector": provides,
            "roles_at_event": roles,
            "value_chain_positions": positions or None,
        },
    )
    logger.info(
        "Extracted %s: roles=%s, positions=%d, needs=%s, provides=%s",
        profile.name,
        roles,
        len(positions),
        needs.primary_need[:40],
        provides.primary_capability[:40],
    )
    return updated


def clear_cache() -> None:
    _cache.clear()
