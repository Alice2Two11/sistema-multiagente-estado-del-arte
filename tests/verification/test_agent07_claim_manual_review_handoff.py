import csv
import io
import json

from src.adapters.agent07c_handoff import prepare_agent07c_input_from_agent07
from src.tools.verification.resolution import resolve_multiple_correction_proposals
from test_multi_proposal_resolution_phase66 import bundle, claim
from test_pipeline_compatibility_handoffs import _handoff_args


def test_claim_level_manual_review_is_not_lost_when_resolution_has_no_manual_flag():
    claim_row = claim(corrections=())
    claim_row["manual_review_required"] = True

    provisional = bundle((), claim_row=claim_row)
    resolution = resolve_multiple_correction_proposals(provisional)

    assert resolution.claim_resolution_plans[0]["manual_review_required"] is False
    assert resolution.claim_resolution_plans[0]["eligible_for_07c"] is False

    draft = {
        "sections": [
            {
                "section_id": "s1",
                "text": "Alpha beta gamma.",
            }
        ]
    }

    prepared = prepare_agent07c_input_from_agent07(
        provisional_bundle=provisional.to_dict(),
        resolution_result=resolution.to_dict(),
        source_draft=draft,
        **_handoff_args(draft, provisional, resolution),
    )

    assert prepared.eligible_claim_ids == ()
    assert prepared.manual_review_claim_ids == ("c1",)

    manifest = json.loads(
        prepared.artifact_payloads[
            "verification_traceability_manifest.json"
        ]
    )
    workflow = manifest["workflow_state"]

    assert workflow["pending_manual_review"] is True
    assert workflow["manual_review_claim_ids"] == ["c1"]
    assert workflow["claim_level_manual_review_count"] == 1
    assert workflow["resolution_level_manual_review_count"] == 0

    queue_rows = list(
        csv.DictReader(
            io.StringIO(
                prepared.optional_artifact_payloads[
                    "manual_review_queue.csv"
                ].decode("utf-8")
            )
        )
    )

    assert len(queue_rows) == 1
    assert queue_rows[0]["claim_id"] == "c1"
    assert queue_rows[0]["verdict"] == "partially_supported"
    assert queue_rows[0]["hallucination_risk"] == "medium"
