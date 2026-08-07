from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from src.adapters.evaluation_upstream import (
    ACCEPTED_CORRECTIONS_REVERIFIED,
    NO_ACCEPTED_CORRECTIONS,
    SOURCE_STAGE_AGENT07,
    SOURCE_STAGE_AGENT07C,
    build_agent08_input_from_agent07c,
    build_agent08_input_from_committed_agent07,
    resolve_agent08_upstream_input,
)


FIXTURE = Path(__file__).parent / "fixtures" / "agent07_direct"


def _copy_direct_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "agent07"
    shutil.copytree(FIXTURE, target)
    return target


def test_real_committed_agent07_direct_path_preserves_partial_and_manual_review(tmp_path):
    source = _copy_direct_fixture(tmp_path)
    result = build_agent08_input_from_committed_agent07(
        agent07_directory=source,
        draft_json_path=source / "state_of_art_draft.json",
        draft_markdown_path=source / "state_of_art_draft.md",
    )

    assert result.source_stage == SOURCE_STAGE_AGENT07
    assert result.reverification_performed is False
    assert result.reverification_reason == NO_ACCEPTED_CORRECTIONS
    assert result.upstream_runtime_status == "PARTIAL"
    assert result.claims_verified == 25
    assert result.claims_requiring_manual_review == 12
    assert result.manual_review_claim_ids == (
        "S2_C1", "S2_C2", "S2_C3", "S2_C5", "S2_C6",
        "S3_C2", "S3_C5", "S4_C6",
        "S5_C1", "S5_C4", "S5_C6", "S5_C7",
    )
    rows = {row["claim_id"]: row for row in result.traceability_rows}
    assert rows["S2_C1"]["verdict"] == "NOT_EVALUATED"
    assert rows["S5_C7"]["manual_review_required"] is True
    pending_rows = [
        row for row in result.traceability_rows
        if row["manual_review_required"]
    ]
    assert sum(row["verdict"] == "NOT_EVALUATED" for row in pending_rows) == 8
    assert sum(row["verdict"] == "PARTIALLY_SUPPORTED" for row in pending_rows) == 4
    assert sum(row["hallucination_risk"] == "MEDIUM" for row in pending_rows) == 12
    assert sum(row["proposal_status"] == "DEFERRED" for row in pending_rows) == 12
    assert sum(
        row["comparison_stage_availability"] == "BLOCKED_UPSTREAM"
        for row in pending_rows
    ) == 12
    assert result.provenance["claims_eligible_for_07c"] == 0
    assert result.provenance["correction_applied"] is False
    assert result.provenance["evaluation_ready_emitted"] is False


def test_direct_path_rejects_when_07c_is_required(tmp_path):
    source = _copy_direct_fixture(tmp_path)
    path = source / "multi_proposal_resolution_result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["eligible_for_07c"] = True
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="AGENT08_DIRECT_PATH_FORBIDDEN_07C_REQUIRED"):
        build_agent08_input_from_committed_agent07(
            agent07_directory=source,
            draft_json_path=source / "state_of_art_draft.json",
            draft_markdown_path=source / "state_of_art_draft.md",
        )


def test_direct_path_rejects_accepted_corrections(tmp_path):
    source = _copy_direct_fixture(tmp_path)
    path = source / "provisional_verification_traceability_bundle.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["metrics"]["corrections_accepted_for_07c"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="AGENT08_DIRECT_PATH_FORBIDDEN_ACCEPTED_CORRECTIONS"):
        build_agent08_input_from_committed_agent07(
            agent07_directory=source,
            draft_json_path=source / "state_of_art_draft.json",
            draft_markdown_path=source / "state_of_art_draft.md",
        )


def test_direct_path_rejects_manual_review_count_mismatch(tmp_path):
    source = _copy_direct_fixture(tmp_path)
    path = source / "provisional_verification_traceability_bundle.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["metrics"]["claims_requiring_manual_review"] = 11
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="AGENT08_AGENT07_MANUAL_REVIEW_COUNT_MISMATCH"):
        build_agent08_input_from_committed_agent07(
            agent07_directory=source,
            draft_json_path=source / "state_of_art_draft.json",
            draft_markdown_path=source / "state_of_art_draft.md",
        )


def test_resolver_uses_direct_path_when_07c_outputs_do_not_exist(tmp_path):
    source = _copy_direct_fixture(tmp_path)
    empty_07c = tmp_path / "agent07c"
    empty_07c.mkdir()
    result = resolve_agent08_upstream_input(
        agent07_directory=source,
        agent07c_directory=empty_07c,
        draft_json_path=source / "state_of_art_draft.json",
        draft_markdown_path=source / "state_of_art_draft.md",
    )
    assert result.source_stage == SOURCE_STAGE_AGENT07
    assert result.reverification_performed is False


def test_agent07c_path_remains_supported(tmp_path):
    directory = tmp_path / "agent07c"
    directory.mkdir()
    draft = json.loads((FIXTURE / "state_of_art_draft.json").read_text(encoding="utf-8"))
    for section in draft["sections"]:
        section["verified_text"] = section.pop("draft_text")
    (directory / "verified_state_of_art_EVALUATION_READY.json").write_text(
        json.dumps({
            **draft,
            "status": "verified_final_after_recheck",
            "verification_summary": {"total_claims": 25, "approved_for_final_evaluation": True},
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    shutil.copy2(
        FIXTURE / "state_of_art_draft.md",
        directory / "verified_state_of_art_EVALUATION_READY.md",
    )
    (directory / "post_correction_recheck_validation_report.json").write_text(
        json.dumps({"validation_ok": True, "approved_for_final_evaluation": True}),
        encoding="utf-8",
    )
    (directory / "post_correction_recheck_manifest.json").write_text(
        json.dumps({
            "workflow_state": {
                "post_correction_recheck_completed": True,
                "all_applied_corrections_rechecked": True,
                "approved_for_final_evaluation": True,
                "evaluation_ready_copy_created": True,
                "final_status": "verified_final_after_recheck",
            },
            "counts": {"pending_manual_review": 0},
        }),
        encoding="utf-8",
    )

    result = build_agent08_input_from_agent07c(agent07c_directory=directory)
    assert result.source_stage == SOURCE_STAGE_AGENT07C
    assert result.reverification_performed is True
    assert result.reverification_reason == ACCEPTED_CORRECTIONS_REVERIFIED
    assert result.claims_requiring_manual_review == 0
