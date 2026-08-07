from copy import deepcopy
import pytest

from src.tools.verification.corrections import fingerprint_text
from src.tools.verification.validation import validate_correction_proposal_contract
from tests.verification.test_terminal_contracts_phase650t import localized_proposal, empty


def assert_error(data, code):
    with pytest.raises(ValueError, match=code):
        validate_correction_proposal_contract(data)


def test_original_text_fingerprint_mismatch():
    d = localized_proposal().to_dict()
    d["original_text"] = "11 ms"
    assert_error(d, "CORRECTION_PROPOSAL_ORIGINAL_TEXT_FINGERPRINT_MISMATCH")


def test_target_text_altered_with_old_fingerprint():
    d = localized_proposal().to_dict()
    d["target_text"] = "11"
    assert_error(d, "CORRECTION_PROPOSAL_TARGET_TEXT_FINGERPRINT_MISMATCH")


def test_target_text_fingerprint_altered():
    d = localized_proposal().to_dict()
    d["target_text_fingerprint"] = "0" * 64
    assert_error(d, "CORRECTION_PROPOSAL_TARGET_TEXT_FINGERPRINT_MISMATCH")


def test_target_span_text_altered():
    d = localized_proposal().to_dict()
    d["target_span_in_claim"]["text"] = "11"
    assert_error(d, "CORRECTION_PROPOSAL_TARGET_SPAN_TEXT_MISMATCH")


def test_target_span_base_fingerprint_altered():
    d = localized_proposal().to_dict()
    d["target_span_in_claim"]["base_text_fingerprint"] = "0" * 64
    assert_error(d, "CORRECTION_PROPOSAL_TARGET_SPAN_BASE_FINGERPRINT_MISMATCH")


def test_claim_span_text_altered():
    d = localized_proposal().to_dict()
    d["claim_span_in_section"]["text"] = "11 ms"
    assert_error(d, "CORRECTION_PROPOSAL_CLAIM_SPAN_TEXT_MISMATCH")


def test_claim_span_base_fingerprint_altered():
    d = localized_proposal().to_dict()
    d["claim_span_in_section"]["base_text_fingerprint"] = "0" * 64
    assert_error(d, "CORRECTION_PROPOSAL_CLAIM_SPAN_BASE_FINGERPRINT_MISMATCH")


def test_target_span_length_incompatible():
    d = localized_proposal().to_dict()
    d["target_span_in_claim"]["end"] = 3
    assert_error(d, "CORRECTION_PROPOSAL_SPAN_LENGTH_MISMATCH:target_span_in_claim")


def test_claim_span_length_incompatible():
    d = localized_proposal().to_dict()
    d["claim_span_in_section"]["end"] = 4
    assert_error(d, "CORRECTION_PROPOSAL_SPAN_LENGTH_MISMATCH:claim_span_in_section")


def test_proposed_claim_text_altered():
    d = localized_proposal().to_dict()
    d["proposed_claim_text"] = "13 ms"
    assert_error(d, "CORRECTION_PROPOSAL_PROPOSED_CLAIM_RECONSTRUCTION_MISMATCH")


def test_localized_reconstruction_valid():
    out = validate_correction_proposal_contract(localized_proposal().to_dict())
    assert out["proposed_claim_text"] == "12 ms"


def test_empty_target_text_fingerprint_altered():
    d = empty("NO_CORRECTION", "NOT_PROPOSED").to_dict()
    d["target_text_fingerprint"] = "0" * 64
    assert_error(d, "CORRECTION_PROPOSAL_TARGET_TEXT_FINGERPRINT_MISMATCH")


def test_empty_proposed_claim_differs_from_original():
    d = empty("NO_CORRECTION", "NOT_PROPOSED").to_dict()
    d["proposed_claim_text"] = "different"
    assert_error(d, "CORRECTION_PROPOSAL_PROPOSED_CLAIM_RECONSTRUCTION_MISMATCH")


def test_empty_valid():
    out = validate_correction_proposal_contract(empty("NO_CORRECTION", "NOT_PROPOSED").to_dict())
    assert out["target_text"] == ""
    assert out["target_text_fingerprint"] == fingerprint_text("")
    assert out["proposed_claim_text"] == out["original_text"]


def test_localized_valid():
    out = validate_correction_proposal_contract(localized_proposal().to_dict())
    assert out["target_span_in_claim"]["text"] == out["target_text"]
