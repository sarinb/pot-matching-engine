"""Unit tests for the deterministic domain model."""

from src.matching.domain_model import (
    ALL_PROBLEM_DOMAINS,
    POSITION_TAGS,
    ROLE_TRANSACTION_MATRIX,
    VALUE_CHAINS,
    best_chain_score,
    best_transaction_type,
    get_transaction_type,
    non_obvious_tag_score,
    position_tags,
    shared_problem_domains,
    stage_compatibility,
    value_chain_adjacency_score,
)


class TestValueChains:
    def test_chain_count(self):
        assert len(VALUE_CHAINS) == 4

    def test_adjacency_same_chain_immediate(self):
        s = value_chain_adjacency_score(
            "tokenized_securities", "issuance",
            "tokenized_securities", "custody",
        )
        assert s == 1.0

    def test_adjacency_same_chain_far(self):
        s = value_chain_adjacency_score(
            "tokenized_securities", "issuance",
            "tokenized_securities", "compliance_audit",
        )
        assert s == 0.3

    def test_adjacency_same_position(self):
        s = value_chain_adjacency_score(
            "tokenized_securities", "custody",
            "tokenized_securities", "custody",
        )
        assert s == 0.2

    def test_cross_chain_high_overlap(self):
        s = value_chain_adjacency_score(
            "tokenized_securities", "compliance_audit",
            "defi_infrastructure", "compliance",
        )
        assert s >= 0.5

    def test_cross_chain_low_overlap(self):
        s = value_chain_adjacency_score(
            "tokenized_securities", "issuance",
            "defi_infrastructure", "wallet_access",
        )
        assert s <= 0.5


class TestBestChainScore:
    def test_single_positions(self):
        s = best_chain_score(
            [("tokenized_securities", "issuance")],
            [("tokenized_securities", "custody")],
        )
        assert s == 1.0

    def test_empty_positions(self):
        s = best_chain_score([], [("tokenized_securities", "issuance")])
        assert s == 0.3


class TestTransactionMatrix:
    def test_investment(self):
        assert get_transaction_type("deploying_capital", "raising_capital") == "investment"

    def test_no_transaction(self):
        assert get_transaction_type("other", "other") is None

    def test_multi_role_best(self):
        tx = best_transaction_type(
            ["deploying_capital", "exploring_partnerships"],
            ["raising_capital", "seeking_technology"],
        )
        assert tx is not None

    def test_multi_role_no_match(self):
        tx = best_transaction_type(["other"], ["other"])
        assert tx is None


class TestStageCompatibility:
    def test_known_rule(self):
        s = stage_compatibility("investment", "sovereign_fund", "series_b")
        assert s == 0.9

    def test_unknown_combination(self):
        s = stage_compatibility("investment", "unknown_a", "unknown_b")
        assert s == 0.5

    def test_null_stage(self):
        s = stage_compatibility("investment", None, "series_b")
        assert s == 0.5


class TestPositionTags:
    def test_tags_exist(self):
        tags = position_tags("tokenized_securities", "issuance")
        assert "regulatory_compliance" in tags
        assert len(tags) >= 2

    def test_all_positions_have_tags(self):
        for chain, positions in VALUE_CHAINS.items():
            for pos in positions:
                key = f"{chain}/{pos}"
                assert key in POSITION_TAGS, f"Missing tags for {key}"

    def test_all_domains_non_empty(self):
        assert len(ALL_PROBLEM_DOMAINS) >= 10


class TestNonObviousTagScore:
    def test_same_chain_shared(self):
        score, shared = non_obvious_tag_score(
            [("tokenized_securities", "issuance")],
            [("tokenized_securities", "custody")],
            None, None,
        )
        assert len(shared) >= 1
        assert score > 0

    def test_cross_sector_boost(self):
        score_same, _ = non_obvious_tag_score(
            [("tokenized_securities", "compliance_audit")],
            [("defi_infrastructure", "compliance")],
            "tokenized_securities", "tokenized_securities",
        )
        score_diff, _ = non_obvious_tag_score(
            [("tokenized_securities", "compliance_audit")],
            [("defi_infrastructure", "compliance")],
            "tokenized_securities", "defi_infrastructure",
        )
        assert score_diff >= score_same

    def test_no_shared_tags(self):
        score, shared = non_obvious_tag_score(
            [("tokenized_securities", "issuance")],
            [("defi_infrastructure", "wallet_access")],
            "a", "b",
        )
        if not shared:
            assert score == 0.0


class TestSharedProblemDomains:
    def test_overlap(self):
        shared = shared_problem_domains(
            [("tokenized_securities", "compliance_audit")],
            [("defi_infrastructure", "compliance")],
        )
        assert "regulatory_compliance" in shared
