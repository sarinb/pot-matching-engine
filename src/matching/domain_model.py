"""Deterministic domain model — the core intellectual asset.

Four layers encoding industry expertise. No LLM calls. Fully unit-testable.
Designed as the white-label customization surface: swap value chains for a
different event vertical.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Layer 1 — Value Chain Map
# ---------------------------------------------------------------------------

VALUE_CHAINS: dict[str, list[str]] = {
    "tokenized_securities": [
        "issuance",
        "custody",
        "settlement",
        "distribution",
        "compliance_audit",
    ],
    "defi_infrastructure": [
        "protocol",
        "scaling_l1_l2",
        "interoperability",
        "wallet_access",
        "compliance",
    ],
    "institutional_adoption": [
        "regulatory_framework",
        "sandbox",
        "pilot",
        "production",
        "scale",
    ],
    "capital_markets": [
        "fund_formation",
        "deal_sourcing",
        "due_diligence",
        "portfolio_management",
        "exit",
    ],
}

_POSITION_INDEX: dict[str, dict[str, int]] = {
    chain: {pos: idx for idx, pos in enumerate(positions)}
    for chain, positions in VALUE_CHAINS.items()
}


# ---------------------------------------------------------------------------
# Layer 1b — Problem Domain Tags per Position
# ---------------------------------------------------------------------------

POSITION_TAGS: dict[str, set[str]] = {
    # tokenized_securities
    "tokenized_securities/issuance": {
        "market_structure", "regulatory_compliance", "asset_origination",
    },
    "tokenized_securities/custody": {
        "institutional_trust", "asset_safeguarding", "regulatory_compliance",
    },
    "tokenized_securities/settlement": {
        "settlement_finality", "cross_border_interop", "institutional_trust",
    },
    "tokenized_securities/distribution": {
        "market_structure", "cross_border_interop", "investor_access",
    },
    "tokenized_securities/compliance_audit": {
        "regulatory_compliance", "identity_access", "institutional_trust",
    },
    # defi_infrastructure
    "defi_infrastructure/protocol": {
        "protocol_design", "settlement_finality", "throughput_scaling",
    },
    "defi_infrastructure/scaling_l1_l2": {
        "throughput_scaling", "settlement_finality", "institutional_trust",
    },
    "defi_infrastructure/interoperability": {
        "cross_border_interop", "settlement_finality", "protocol_design",
    },
    "defi_infrastructure/wallet_access": {
        "identity_access", "investor_access", "user_onboarding",
    },
    "defi_infrastructure/compliance": {
        "regulatory_compliance", "identity_access", "institutional_trust",
    },
    # institutional_adoption
    "institutional_adoption/regulatory_framework": {
        "regulatory_compliance", "market_structure", "institutional_trust",
    },
    "institutional_adoption/sandbox": {
        "regulatory_compliance", "market_validation", "institutional_trust",
    },
    "institutional_adoption/pilot": {
        "market_validation", "institutional_trust", "throughput_scaling",
    },
    "institutional_adoption/production": {
        "institutional_trust", "settlement_finality", "asset_safeguarding",
    },
    "institutional_adoption/scale": {
        "cross_border_interop", "throughput_scaling", "investor_access",
    },
    # capital_markets
    "capital_markets/fund_formation": {
        "market_structure", "regulatory_compliance", "investor_access",
    },
    "capital_markets/deal_sourcing": {
        "market_validation", "investor_access", "asset_origination",
    },
    "capital_markets/due_diligence": {
        "institutional_trust", "regulatory_compliance", "market_validation",
    },
    "capital_markets/portfolio_management": {
        "asset_safeguarding", "institutional_trust", "cross_border_interop",
    },
    "capital_markets/exit": {
        "market_structure", "investor_access", "cross_border_interop",
    },
}

ALL_PROBLEM_DOMAINS: set[str] = set()
for _tags in POSITION_TAGS.values():
    ALL_PROBLEM_DOMAINS.update(_tags)

_DOMAIN_TO_POSITIONS: dict[str, set[str]] = {}
for _pos, _tags in POSITION_TAGS.items():
    for _tag in _tags:
        _DOMAIN_TO_POSITIONS.setdefault(_tag, set()).add(_pos)


def position_tags(chain: str, position: str) -> set[str]:
    return POSITION_TAGS.get(f"{chain}/{position}", set())


def shared_problem_domains(
    positions_a: list[tuple[str, str]],
    positions_b: list[tuple[str, str]],
) -> set[str]:
    tags_a: set[str] = set()
    for chain, pos in positions_a:
        tags_a.update(position_tags(chain, pos))
    tags_b: set[str] = set()
    for chain, pos in positions_b:
        tags_b.update(position_tags(chain, pos))
    return tags_a & tags_b


def non_obvious_tag_score(
    positions_a: list[tuple[str, str]],
    positions_b: list[tuple[str, str]],
    sector_a: str | None,
    sector_b: str | None,
) -> tuple[float, set[str]]:
    """Score non-obvious connection potential from shared problem domains.

    Returns (score, shared_tags).  Higher when positions share problem
    domains across different sectors.
    """
    shared = shared_problem_domains(positions_a, positions_b)
    if not shared:
        return 0.0, shared

    tags_a: set[str] = set()
    for chain, pos in positions_a:
        tags_a.update(position_tags(chain, pos))
    tags_b: set[str] = set()
    for chain, pos in positions_b:
        tags_b.update(position_tags(chain, pos))

    union = tags_a | tags_b
    jaccard = len(shared) / len(union) if union else 0.0
    base = jaccard

    different_sectors = (
        sector_a is not None
        and sector_b is not None
        and sector_a != sector_b
    )
    if different_sectors:
        base = min(base * 1.5, 1.0)

    return base, shared


# ---------------------------------------------------------------------------
# Value chain adjacency
# ---------------------------------------------------------------------------

def value_chain_adjacency_score(
    chain_a: str, pos_a: str, chain_b: str, pos_b: str,
) -> float:
    if chain_a != chain_b:
        tags_a = position_tags(chain_a, pos_a)
        tags_b = position_tags(chain_b, pos_b)
        if not tags_a or not tags_b:
            return 0.3
        overlap = len(tags_a & tags_b)
        if overlap >= 2:
            return 0.8
        if overlap == 1:
            return 0.5
        return 0.2

    positions = _POSITION_INDEX.get(chain_a)
    if positions is None or pos_a not in positions or pos_b not in positions:
        return 0.3
    distance = abs(positions[pos_a] - positions[pos_b])
    if distance == 0:
        return 0.2
    if distance == 1:
        return 1.0
    if distance == 2:
        return 0.6
    return 0.3


def best_chain_score(
    positions_a: list[tuple[str, str]],
    positions_b: list[tuple[str, str]],
) -> float:
    if not positions_a or not positions_b:
        return 0.3
    best = 0.0
    for chain_a, pos_a in positions_a:
        for chain_b, pos_b in positions_b:
            s = value_chain_adjacency_score(chain_a, pos_a, chain_b, pos_b)
            if s > best:
                best = s
    return best


# ---------------------------------------------------------------------------
# Layer 2 — Role-Transaction Compatibility Matrix
# ---------------------------------------------------------------------------

ROLE_TRANSACTION_MATRIX: dict[tuple[str, str], str | None] = {
    ("deploying_capital", "deploying_capital"): "co_investment",
    ("deploying_capital", "raising_capital"): "investment",
    ("deploying_capital", "exploring_partnerships"): None,
    ("deploying_capital", "seeking_technology"): "investment",
    ("deploying_capital", "regulatory_policy"): "policy_dialogue",
    ("deploying_capital", "media_content"): None,
    ("deploying_capital", "other"): None,

    ("raising_capital", "deploying_capital"): "fundraising",
    ("raising_capital", "raising_capital"): None,
    ("raising_capital", "exploring_partnerships"): "gtm_partnership",
    ("raising_capital", "seeking_technology"): "tech_partnership",
    ("raising_capital", "regulatory_policy"): "sandbox_candidacy",
    ("raising_capital", "media_content"): "media_exposure",
    ("raising_capital", "other"): None,

    ("exploring_partnerships", "deploying_capital"): None,
    ("exploring_partnerships", "raising_capital"): "gtm_partnership",
    ("exploring_partnerships", "exploring_partnerships"): "partnership",
    ("exploring_partnerships", "seeking_technology"): "integration",
    ("exploring_partnerships", "regulatory_policy"): "sandbox_participation",
    ("exploring_partnerships", "media_content"): None,
    ("exploring_partnerships", "other"): None,

    ("seeking_technology", "deploying_capital"): "investment_pitch",
    ("seeking_technology", "raising_capital"): "tech_evaluation",
    ("seeking_technology", "exploring_partnerships"): "integration",
    ("seeking_technology", "seeking_technology"): "technical_collaboration",
    ("seeking_technology", "regulatory_policy"): "sandbox_evaluation",
    ("seeking_technology", "media_content"): None,
    ("seeking_technology", "other"): None,

    ("regulatory_policy", "deploying_capital"): "policy_dialogue",
    ("regulatory_policy", "raising_capital"): "sandbox_candidates",
    ("regulatory_policy", "exploring_partnerships"): "sandbox_participation",
    ("regulatory_policy", "seeking_technology"): "sandbox_evaluation",
    ("regulatory_policy", "regulatory_policy"): "policy_coordination",
    ("regulatory_policy", "media_content"): "content_collaboration",
    ("regulatory_policy", "other"): None,

    ("media_content", "deploying_capital"): None,
    ("media_content", "raising_capital"): "media_exposure",
    ("media_content", "exploring_partnerships"): None,
    ("media_content", "seeking_technology"): None,
    ("media_content", "regulatory_policy"): "content_collaboration",
    ("media_content", "media_content"): "content_collaboration",
    ("media_content", "other"): None,

    ("other", "deploying_capital"): None,
    ("other", "raising_capital"): None,
    ("other", "exploring_partnerships"): None,
    ("other", "seeking_technology"): None,
    ("other", "regulatory_policy"): None,
    ("other", "media_content"): None,
    ("other", "other"): None,
}


def get_transaction_type(role_a: str, role_b: str) -> str | None:
    return ROLE_TRANSACTION_MATRIX.get((role_a, role_b))


def best_transaction_type(
    roles_a: list[str], roles_b: list[str],
) -> str | None:
    """Find the highest-value transaction type across all role combinations."""
    for ra in roles_a:
        for rb in roles_b:
            tx = ROLE_TRANSACTION_MATRIX.get((ra, rb))
            if tx is not None:
                return tx
    return None


# ---------------------------------------------------------------------------
# Layer 3 — Stage Compatibility Rules
# ---------------------------------------------------------------------------

STAGE_RULES: dict[str, dict[tuple[str, str], float]] = {
    "investment": {
        ("sovereign_fund", "series_b"): 0.9,
        ("sovereign_fund", "pre_product"): 0.1,
        ("growth_vc", "series_b"): 0.95,
        ("growth_vc", "seed"): 0.3,
        ("seed_fund", "seed"): 0.9,
        ("seed_fund", "series_b"): 0.4,
    },
    "fundraising": {
        ("series_b", "sovereign_fund"): 0.9,
        ("series_b", "growth_vc"): 0.95,
        ("seed", "seed_fund"): 0.9,
        ("seed", "growth_vc"): 0.3,
    },
    "co_investment": {
        ("sovereign_fund", "growth_vc"): 0.8,
        ("growth_vc", "sovereign_fund"): 0.8,
        ("growth_vc", "growth_vc"): 0.7,
    },
    "regulatory": {
        ("central_bank", "growth_startup"): 0.8,
        ("central_bank", "pre_product"): 0.4,
    },
    "partnership": {
        ("growth_startup", "growth_startup"): 0.7,
    },
}


def stage_compatibility(
    tx_type: str, stage_a: str | None, stage_b: str | None,
) -> float:
    if not stage_a or not stage_b:
        return 0.5
    rules = STAGE_RULES.get(tx_type, {})
    return rules.get((stage_a, stage_b), 0.5)


# ---------------------------------------------------------------------------
# Deterministic inference helpers
# ---------------------------------------------------------------------------

_TITLE_SENIORITY: dict[str, float] = {
    "ceo": 0.95, "cfo": 0.9, "cto": 0.85, "coo": 0.85, "cio": 0.85,
    "founder": 0.9, "co-founder": 0.9, "president": 0.9,
    "managing director": 0.85, "general partner": 0.9, "partner": 0.8,
    "director": 0.75, "head": 0.75, "vp": 0.7, "vice president": 0.7,
    "senior": 0.6, "manager": 0.5, "analyst": 0.3,
}


def infer_mandate_score(title: str) -> float:
    title_lower = title.lower()
    for keyword, score in _TITLE_SENIORITY.items():
        if keyword in title_lower:
            return score
    return 0.5


def infer_maturity_score(
    key_facts: list[str] | None, stage: str | None,
) -> float:
    score = 0.5
    if stage:
        stage_l = stage.lower()
        if any(s in stage_l for s in ("series_b", "series_c", "growth", "scale")):
            score += 0.2
        elif any(s in stage_l for s in ("series_a", "seed")):
            score += 0.1
        elif "pre" in stage_l:
            score -= 0.1

    if key_facts:
        joined = " ".join(key_facts).lower()
        maturity_signals = ["live", "customer", "deploy", "production", "bank", "partner"]
        hits = sum(1 for s in maturity_signals if s in joined)
        score += min(hits * 0.05, 0.2)

    return min(max(score, 0.0), 1.0)


def infer_capability(title: str, company_desc: str | None) -> str:
    parts = [title]
    if company_desc:
        parts.append(company_desc)
    text = " ".join(parts).lower()

    if any(w in text for w in ("fund", "invest", "capital", "wealth", "allocat")):
        return "capital_deployment"
    if any(w in text for w in ("custod", "settlement", "clearing")):
        return "custody_settlement"
    if any(w in text for w in ("l2", "layer", "scaling", "protocol", "infrastructure")):
        return "blockchain_infrastructure"
    if any(w in text for w in ("regulat", "policy", "compliance", "central bank", "cbdc")):
        return "regulatory_framework"
    if any(w in text for w in ("venture", "vc", "gp", "general partner")):
        return "venture_investment"
    return "technology_services"


def infer_audience(title: str, company_desc: str | None) -> list[str]:
    text = f"{title} {company_desc or ''}".lower()
    audiences: list[str] = []
    if any(w in text for w in ("bank", "institution", "sovereign")):
        audiences.append("institutional_investors")
    if any(w in text for w in ("startup", "founder", "ceo")):
        audiences.append("startups")
    if any(w in text for w in ("regulat", "policy", "government")):
        audiences.append("regulators")
    return audiences or ["general"]


def infer_geographic_reach(company_desc: str | None) -> list[str]:
    if not company_desc:
        return ["global"]
    text = company_desc.lower()
    regions: list[str] = []
    if any(w in text for w in ("europe", "eu", "mica", "german", "french")):
        regions.append("europe")
    if any(w in text for w in ("middle east", "abu dhabi", "dubai", "saudi")):
        regions.append("middle_east")
    if any(w in text for w in ("asia", "apac", "singapore", "japan", "korea")):
        regions.append("apac")
    if any(w in text for w in ("us", "america", "new york")):
        regions.append("north_america")
    return regions or ["global"]
