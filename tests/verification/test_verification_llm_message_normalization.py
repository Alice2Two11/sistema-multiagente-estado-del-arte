from __future__ import annotations

import json

import pytest

from src.agents.verification_agent import VerificationAgent
from src.config.verification_policy_config import get_verification_input_policy
from src.tools.verification.prompting import parse_verification_response

try:  # Uses the real LangChain class when the dependency is available.
    from langchain_core.messages import AIMessage  # type: ignore
except ModuleNotFoundError:  # Protocol-compatible test double for the isolated suite.
    class AIMessage:  # noqa: D101
        def __init__(self, content: object) -> None:
            self.content = content


class SimulatedChatOpenAI:
    """ChatOpenAI boundary double returning a LangChain-style AIMessage."""

    def __init__(self, content: str) -> None:
        self._response = AIMessage(content=content)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self._response


def _s2_c1_context() -> dict:
    return {
        "claim_id": "S2_C1",
        "claim_id_origin": "AGENT06",
        "section_id": "S2",
        "section_title": "Modelos ANN para pronóstico solar",
        "claim_text": "El modelo ANN obtuvo un RMSE de 1.34 MJ·m−2.",
        "claim_type": "QUANTITATIVE",
        "verification_intensity": "STRICT",
        "supporting_citations": (
            {"source_filename": "paper_ann.pdf", "chunk_id": "chunk_0022"},
        ),
        "inherited_evidence_assessment": {
            "evidence_rows": (
                {
                    "source_filename": "paper_ann.pdf",
                    "chunk_id": "chunk_0022",
                    "text": "El modelo ANN obtuvo un RMSE de 1.34 MJ·m−2.",
                },
            ),
            "additional_evidence_rows": (),
        },
        "retrieval_result": {"selected_candidates": (), "rounds_executed": 0},
        "deterministic_validation": {
            "citation_valid": True,
            "document_identity_valid": True,
            "authorization_valid": True,
            "numeric_pairs_valid": True,
            "deterministic_issue_codes": (),
        },
        "allowed_source_pairs": (("paper_ann.pdf", "chunk_0022"),),
        "policy": get_verification_input_policy(),
        "attempt_context": {
            "remaining_retrieval_requests": 0,
            "correction_localized": False,
        },
    }


def _valid_s2_c1_response() -> dict:
    return {
        "claim_id": "S2_C1",
        "verdict": "SUPPORTED",
        "support_level": "STRONG",
        "evidence_ids_used": ["E01"],
        "evidence_ids_rejected": [],
        "rationale": "La evidencia autorizada respalda el valor cuantitativo.",
        "contradiction_type": "NONE",
        "contradiction_evidence_ids": [],
        "numeric_assessment": "SUPPORTED",
        "attribution_assessment": "NOT_APPLICABLE",
        "extrapolation_assessment": "WITHIN_EVIDENCE_SCOPE",
        "confidence": "LOW",
        "additional_retrieval_needed": False,
        "llm_correction_recommendation": False,
        "manual_review_required": False,
        "reason_codes": [],
    }


def test_parse_verification_response_accepts_string() -> None:
    payload = {"claim_id": "S2_C1"}
    assert parse_verification_response(json.dumps(payload)) == payload


def test_parse_verification_response_accepts_mapping() -> None:
    payload = {"claim_id": "S2_C1"}
    assert parse_verification_response(payload) == payload


def test_parse_verification_response_accepts_ai_message_content() -> None:
    payload = {"claim_id": "S2_C1"}
    message = AIMessage(content=json.dumps(payload))
    assert parse_verification_response(message) == payload


def test_parse_verification_response_rejects_truly_empty_content() -> None:
    with pytest.raises(ValueError, match="^LLM_RESPONSE_EMPTY$"):
        parse_verification_response(AIMessage(content="   "))


def test_parse_verification_response_rejects_non_json_content() -> None:
    with pytest.raises(ValueError, match="^LLM_RESPONSE_NOT_PURE_JSON_OBJECT$"):
        parse_verification_response(AIMessage(content="not json"))


def test_ai_message_with_incomplete_json_reaches_schema_validation() -> None:
    llm = SimulatedChatOpenAI(json.dumps({"claim_id": "S2_C1"}))
    result = VerificationAgent(llm=llm).verify_claim(_s2_c1_context())

    assert llm.calls == 3
    assert result.raw_attempts
    assert all(row["parse_status"] == "SCHEMA_INVALID" for row in result.raw_attempts)
    assert all("LLM_RESPONSE_EMPTY" not in row["validation_errors"] for row in result.raw_attempts)
    assert result.tool_usage["schema_validation_attempts"] == 3


def test_real_s2_c1_with_simulated_chatopenai_ai_message_reaches_scientific_judgment() -> None:
    llm = SimulatedChatOpenAI(json.dumps(_valid_s2_c1_response(), ensure_ascii=False))
    result = VerificationAgent(llm=llm).verify_claim(_s2_c1_context())

    assert llm.calls == 1
    assert result.claim_id == "S2_C1"
    assert result.scientific_verdict == "SUPPORTED"
    assert result.scientific_judgment_status == "COMPLETED"
    assert result.tool_usage["schema_validation_attempts"] == 1
    assert result.tool_usage["scientific_judgment_attempts"] == 1
    assert result.raw_attempts[0]["parse_status"] == "PARSED"
    assert isinstance(result.raw_attempts[0]["raw_text"], str)
    assert "LLM_RESPONSE_EMPTY" not in result.raw_attempts[0]["validation_errors"]


def test_partial_support_fallback_marks_manual_review_consistently() -> None:
    response = _valid_s2_c1_response()
    response.update({
        "verdict": "PARTIALLY_SUPPORTED",
        "support_level": "PARTIAL",
        "reason_codes": ["PARTIAL_EVIDENCE"],
        "manual_review_required": False,
        "llm_correction_recommendation": False,
    })
    llm = SimulatedChatOpenAI(json.dumps(response, ensure_ascii=False))

    result = VerificationAgent(llm=llm).verify_claim(_s2_c1_context())

    assert result.scientific_verdict == "PARTIALLY_SUPPORTED"
    assert result.final_correction_eligibility == "MANUAL_REVIEW_REQUIRED"
    assert result.manual_review_required is True

    # The exact terminal contract that failed for S3_C2 must now accept this row.
    from src.tools.verification.validation import validate_claim_verification_result_contract
    from dataclasses import asdict
    validated = validate_claim_verification_result_contract(asdict(result))
    assert validated["manual_review_required"] is True
