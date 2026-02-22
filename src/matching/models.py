"""Pydantic v2 data models — the data contracts flowing through the system."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TicketType = Literal["delegate", "sponsor", "speaker", "vip"]

RoleAtEvent = Literal[
    "deploying_capital",
    "raising_capital",
    "exploring_partnerships",
    "seeking_technology",
    "regulatory_policy",
    "media_content",
    "other",
]

ValueChainPosition = tuple[str, str]  # (chain_name, position)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug)
    return slug


# ---------------------------------------------------------------------------
# Structured vectors
# ---------------------------------------------------------------------------

class NeedsVector(BaseModel):
    primary_need: str
    need_description: str = ""
    target_counterparty_type: str = ""
    urgency: Literal["active", "exploring", "passive"] = "exploring"
    constraints: list[str] = Field(default_factory=list)


class ProvidesVector(BaseModel):
    primary_capability: str
    capability_description: str = ""
    evidence: list[str] = Field(default_factory=list)
    geographic_reach: list[str] = Field(default_factory=list)
    audience_access: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Attendee profile
# ---------------------------------------------------------------------------

class AttendeeProfile(BaseModel):
    id: str | None = None
    name: str
    title: str
    company: str
    product: str | None = None
    company_description: str | None = None
    ticket_type: TicketType = "delegate"
    role_at_event: RoleAtEvent | None = None
    roles_at_event: list[RoleAtEvent] | None = None
    looking_for: str | None = None
    stated_goal: str | None = None
    sector: str | None = None
    stage: str | None = None
    funding_raised: str | None = None
    key_facts: list[str] = Field(default_factory=list)

    needs_vector: NeedsVector | None = None
    provides_vector: ProvidesVector | None = None
    value_chain_positions: list[ValueChainPosition] | None = None

    @model_validator(mode="after")
    def _normalize_fields(self) -> AttendeeProfile:
        if not self.id:
            self.id = _slugify(self.name)
        if not self.stated_goal and self.looking_for:
            self.stated_goal = self.looking_for
        elif not self.stated_goal:
            self.stated_goal = ""
        if not self.company_description and self.product:
            self.company_description = self.product
        if self.roles_at_event is None and self.role_at_event is not None:
            self.roles_at_event = [self.role_at_event]
        return self


# ---------------------------------------------------------------------------
# Scoring / output types
# ---------------------------------------------------------------------------

class DimensionScores(BaseModel):
    complementarity: float = 0.0
    transaction_readiness: float = 0.0
    non_obvious: float = 0.0


class DimensionExplanations(BaseModel):
    complementarity: str = ""
    transaction_readiness: str = ""
    non_obvious: str = ""


class ScoredPair(BaseModel):
    attendee_a_id: str
    attendee_b_id: str
    scores: DimensionScores
    composite: float = 0.0
    transaction_type: str | None = None


class MatchResult(BaseModel):
    pair: ScoredPair
    explanation: str = ""
    dimension_explanations: DimensionExplanations = Field(
        default_factory=DimensionExplanations,
    )


class MatchBriefing(BaseModel):
    attendee_id: str
    attendee_name: str
    matches: list[MatchResult] = Field(default_factory=list)
