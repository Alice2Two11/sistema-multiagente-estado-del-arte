from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from cryptography.fernet import Fernet

from src.agents.extraction_agent import ExtractionAgent
from src.contracts.agent_input import AgentInput, PreviousAttemptSummary
from src.contracts.agent_result import (
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.io.credentials import load_runtime_credential
from src.runtime.extraction_protocol import execute_extraction_runtime_transaction
from src.state.pipeline_state import PipelineIdentity, PipelineState
from src.state.state_store import StateStore
from tests.v16.agent_environment import ExtractionAgentEnvironment
from tests.v16.extraction_agent_doubles import complete_card

ROOT = Path(__file__).resolve().parents[2]
payload = json.loads((ROOT / "preserved_sha256.json").read_text())
if payload.get("status") != "ALL_PROTECTED_FILES_PRESERVED":
    raise AssertionError(
        "preserved_sha256.json status must be ALL_PROTECTED_FILES_PRESERVED"
    )
PRESERVED = payload["files"]


def clone_input(
    agent_input: AgentInput,
    *,
    attempt_number: int,
    previous_result: AgentResult | None = None,
) -> AgentInput:
    payload = agent_input.to_dict()
    payload["attempt_number"] = attempt_number
    if previous_result is not None:
        payload["previous_attempt"] = PreviousAttemptSummary(
            quality_status=previous_result.quality_status.value,
            quality_metrics=previous_result.quality_metrics,
            blocking_warnings=tuple(
                warning.code
                for warning in previous_result.warnings
                if warning.blocking
            ),
            failure_reason_codes=previous_result.failure_reason_codes,
            previous_artifacts=previous_result.output_artifacts,
        ).to_dict()
    return AgentInput.from_dict(payload)


class AlignmentV16Tests(unittest.TestCase):
    def test_required_files_and_forbidden_policy_name(self):
        self.assertTrue((ROOT / "src/config/generation_policy_config.py").is_file())
        self.assertTrue((ROOT / "src/io/credentials.py").is_file())
        self.assertFalse((ROOT / "src/config/generation_config.py").exists())

    def test_preserved_files_are_byte_identical(self):
        for relative, expected in PRESERVED.items():
            with self.subTest(relative=relative):
                observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                self.assertEqual(observed, expected)

    def test_agent_result_failure_codes_roundtrip(self):
        now = datetime.now(timezone.utc).isoformat()
        result = AgentResult(
            execution_status=ExecutionStatus.FAILED,
            quality_status=QualityStatus.REJECTED,
            decision=DecisionInfo(code="FAILED", rationale="dependency"),
            quality_metrics={},
            warnings=(),
            failure_reason_codes=("DEPENDENCY_NOT_FOUND",),
            requested_transition=RequestedTransition(
                action=TransitionAction.HALT_STAGE,
                target_stage=None,
                reason_code="DEPENDENCY_NOT_FOUND",
            ),
            output_artifacts={},
            tool_usage=ToolUsage(),
            attempt_number=1,
            started_at=now,
            completed_at=now,
            error={"type": "FileNotFoundError", "message": "missing"},
        )
        restored = AgentResult.from_dict(result.to_dict())
        self.assertEqual(restored.failure_reason_codes, ("DEPENDENCY_NOT_FOUND",))

    def test_state_store_persists_failure_codes_in_stage_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "pipeline_state.json"
            store = StateStore(state_path)
            now = datetime.now(timezone.utc).isoformat()
            store.initialize(PipelineState(
                identity=PipelineIdentity(
                    experiment_id="exp", run_id="run",
                    created_at=now, updated_at=now, schema_version="1.0",
                ),
                generation_config_snapshot={},
            ))
            transaction = execute_extraction_runtime_transaction(
                store=store,
                attempt_number=1,
                build_execution=lambda: (_ for _ in ()).throw(
                    FileNotFoundError("chunks missing")
                ),
            )
            self.assertEqual(transaction.agent_result.execution_status, ExecutionStatus.FAILED)
            self.assertEqual(transaction.agent_result.failure_reason_codes, ("DEPENDENCY_NOT_FOUND",))
            stage = transaction.committed_state.stages["03_agente_extraccion_kb"]
            self.assertEqual(stage.failure_reason_codes, ("DEPENDENCY_NOT_FOUND",))
            self.assertEqual(
                transaction.committed_state.decision_log[-1].reason_codes,
                ("DEPENDENCY_NOT_FOUND",),
            )

    def test_clean_attempt_one_approves_and_does_not_choose_03b_or_04(self):
        env = ExtractionAgentEnvironment()
        try:
            result = ExtractionAgent(env.dependencies).execute(env.agent_input)
            self.assertEqual(result.execution_status, ExecutionStatus.COMPLETED)
            self.assertIn(result.quality_status, {
                QualityStatus.APPROVED,
                QualityStatus.APPROVED_WITH_WARNINGS,
            })
            self.assertEqual(result.requested_transition.action, TransitionAction.ADVANCE)
            self.assertIsNone(result.requested_transition.target_stage)
        finally:
            env.close()

    def test_attempt_one_invalid_title_is_repaired_before_gate(self):
        original = complete_card("b.pdf", title="no especificado")
        env = ExtractionAgentEnvironment(
            extraction_cards={"b.pdf": original},
            repaired_titles={"b.pdf": "Recovered title"},
        )
        try:
            result = ExtractionAgent(env.dependencies).execute(env.agent_input)
            self.assertIn(result.quality_status, {
                QualityStatus.APPROVED,
                QualityStatus.APPROVED_WITH_WARNINGS,
            })
            repaired = {
                card["source_filename"]: card
                for card in env.read_cards()
            }["b.pdf"]
            self.assertEqual(repaired["title"], "Recovered title")
            for field in (
                "research_problem", "objective", "methods_or_models",
                "evaluation_metrics", "main_results", "evidence",
            ):
                self.assertEqual(repaired[field], original[field])
        finally:
            env.close()

    def test_attempt_one_without_evidence_requests_more_evidence(self):
        card = complete_card("b.pdf")
        card["evidence"] = []
        env = ExtractionAgentEnvironment(
            extraction_cards={"b.pdf": card}
        )
        try:
            result = ExtractionAgent(env.dependencies).execute(env.agent_input)
            self.assertEqual(result.quality_status, QualityStatus.NEEDS_MORE_EVIDENCE)
            self.assertIn("INSUFFICIENT_EVIDENCE", result.failure_reason_codes)
            self.assertEqual(result.requested_transition.action, TransitionAction.RETRY)
        finally:
            env.close()

    def test_attempt_two_partially_usable_can_request_manual_review(self):
        card = complete_card("b.pdf")
        card["evidence"] = []
        env = ExtractionAgentEnvironment(
            extraction_cards={"b.pdf": card},
            repair_errors={"b.pdf": RuntimeError("still incomplete")},
        )
        try:
            first = ExtractionAgent(env.dependencies).execute(env.agent_input)
            payload = clone_input(
                env.agent_input,
                attempt_number=2,
                previous_result=first,
            ).to_dict()
            payload["policy"]["signature"]["extraction_policy"] = {
                "thresholds": {
                    "approval": {"critical_field_coverage": 0.92},
                    "minimum_usable_quality": {"critical_field_coverage": 0.80},
                },
                "manual_review_policy": {
                    "allowed": True,
                    "allowed_reason_codes": [
                        "INSUFFICIENT_EVIDENCE",
                        "MISSING_CRITICAL_FIELDS",
                    ],
                },
            }
            result = ExtractionAgent(env.dependencies).execute(
                AgentInput.from_dict(payload)
            )
            self.assertEqual(
                result.quality_status,
                QualityStatus.APPROVED_PENDING_MANUAL_REVIEW,
            )
            self.assertTrue(result.requested_transition.requires_human_confirmation)
        finally:
            env.close()

    def test_attempt_two_unusable_is_rejected_when_manual_review_disabled(self):
        invalid = [complete_card("b.pdf"), complete_card("b.pdf")]
        env = ExtractionAgentEnvironment(
            extraction_cards={"b.pdf": invalid},
            repair_errors={"b.pdf": RuntimeError("still invalid")},
        )
        try:
            first = ExtractionAgent(env.dependencies).execute(env.agent_input)
            payload = clone_input(
                env.agent_input, attempt_number=2, previous_result=first
            ).to_dict()
            payload["policy"]["signature"]["extraction_policy"] = {
                "thresholds": {
                    "approval": {"critical_field_coverage": 0.92},
                    "minimum_usable_quality": {"critical_field_coverage": 0.80},
                },
                "manual_review_policy": {"allowed": False},
            }
            result = ExtractionAgent(env.dependencies).execute(
                AgentInput.from_dict(payload)
            )
            self.assertEqual(result.quality_status, QualityStatus.REJECTED)
            self.assertEqual(
                result.requested_transition.action, TransitionAction.HALT_STAGE
            )
        finally:
            env.close()

    def test_optional_classification_failure_is_approved_with_warnings(self):
        env = ExtractionAgentEnvironment(
            classification_errors={"b.pdf": RuntimeError("classification failed")}
        )
        try:
            result = ExtractionAgent(env.dependencies).execute(env.agent_input)
            self.assertEqual(result.quality_status, QualityStatus.APPROVED_WITH_WARNINGS)
            self.assertEqual(result.failure_reason_codes, ())
        finally:
            env.close()

    def test_attempt_two_runs_directed_repair(self):
        ambiguous = [complete_card("b.pdf"), complete_card("b.pdf")]
        env = ExtractionAgentEnvironment(
            extraction_cards={"b.pdf": ambiguous},
            repair_cards={"b.pdf": complete_card("b.pdf", title="Recovered")},
        )
        try:
            first = ExtractionAgent(env.dependencies).execute(env.agent_input)
            self.assertEqual(first.quality_status, QualityStatus.NEEDS_REVISION)
            self.assertIn("INVALID_LLM_OUTPUT", first.failure_reason_codes)
            result = ExtractionAgent(env.dependencies).execute(
                clone_input(
                    env.agent_input,
                    attempt_number=2,
                    previous_result=first,
                )
            )
            self.assertEqual(result.execution_status, ExecutionStatus.COMPLETED)
            self.assertGreaterEqual(result.tool_usage.llm_calls, 1)
            self.assertIn(result.quality_status, {
                QualityStatus.APPROVED,
                QualityStatus.APPROVED_WITH_WARNINGS,
            })
            self.assertTrue(env.paths["KB_CSV_PATH"].is_file())
        finally:
            env.close()

    def test_03b_is_not_integrated(self):
        source = (ROOT / "src/agents/extraction_agent.py").read_text().casefold()
        notebook = (ROOT / "03_agente_extraccion_kb_migrado_v16.ipynb").read_text().casefold()
        for token in ("03b", "quantitative_extraction", "structured_quantitative"):
            self.assertNotIn(token, source)
            self.assertNotIn(token, notebook)

    def test_credentials_environment_encrypted_and_no_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret = "secret-value-for-test"
            environment = {"OPENAI_API_KEY": secret}
            self.assertEqual(
                load_runtime_credential(
                    "OPENAI_API_KEY", project_dir=root, environ=environment,
                    colab_userdata_getter=lambda name: "fallback",
                ),
                secret,
            )

            environment = {}
            secrets = root / ".secrets"
            secrets.mkdir(parents=True)
            key = Fernet.generate_key()
            (secrets / "openai_api_key.key").write_bytes(key)
            (secrets / "openai_api_key.enc").write_bytes(
                Fernet(key).encrypt(secret.encode())
            )
            value = load_runtime_credential(
                "OPENAI_API_KEY", project_dir=root, environ=environment,
                colab_userdata_getter=lambda name: "fallback",
            )
            self.assertEqual(value, secret)
            self.assertNotIn(secret, repr(load_runtime_credential))

    def test_credentials_missing_error_is_sanitized(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as captured:
                load_runtime_credential(
                    "OPENAI_API_KEY",
                    project_dir=tmp,
                    environ={},
                    colab_userdata_getter=lambda name: None,
                )
            self.assertNotIn("sk-", str(captured.exception))

    def test_notebook_is_thin_and_real_mode_visible(self):
        notebook = json.loads(
            (ROOT / "03_agente_extraccion_kb_migrado_v16.ipynb").read_text()
        )
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("RUN_REAL_EXTRACTION", source)
        self.assertIn("REAL MODE", source)
        self.assertIn("PRECHECK MODE", source)
        self.assertIn("execute_extraction_runtime_transaction", source)
        for forbidden in (
            "def retrieve_chunks_for_paper",
            "def is_bad_card",
            "def run_title_repair",
            "def build_knowledge_base_rows",
            "Fernet(",
        ):
            self.assertNotIn(forbidden, source)

    def test_notebook_imports_common_names_before_use(self):
        notebook = json.loads(
            (ROOT / "03_agente_extraccion_kb_migrado_v16.ipynb").read_text()
        )
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        compile(source, "notebook_v16", "exec")
        for name, import_text in {
            "json.": "import json",
            "os.": "import os",
            "Path(": "from pathlib import Path",
            "shutil.": "import shutil",
            "subprocess.": "import subprocess",
            "sys.": "import sys",
            "datetime.": "from datetime import datetime",
        }.items():
            if name in source:
                self.assertIn(import_text, source)


if __name__ == "__main__":
    unittest.main()
