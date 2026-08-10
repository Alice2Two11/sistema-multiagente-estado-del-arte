"""Integración de punta a punta: ``claim_uid`` real, minteado en una
ejecución de 06, se propaga a través de una verificación real de 07
(``RETURN``), llega al ``writer_revision_request`` real, y 06 -- en
modo REVISION real -- lo preserva determinísticamente al corregir el
claim señalado, incluso cuando el LLM declara ``identity_action`` por
su cuenta para los DEMÁS claims de la sección.

Reutiliza el mismo flujo real ya probado en
``test_qualitative_correction_keyerror_and_return.py::R06`` (07 real,
06 real, sin dobles del mecanismo en sí) -- la única diferencia es que
aquí el contexto de 06 SÍ lleva ``claim_uid`` desde el principio (como
lo haría un experimento real corrido enteramente bajo el contrato de
identidad estable), y se verifica explícitamente que ese mismo
``claim_uid`` sobrevive intacto en ``revised_draft.json`` tras la
revisión real."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "orchestration"))

import test_verification_stagespec_integration as T  # noqa: E402
import test_writer_verifier_cycle_e2e as E2E  # noqa: E402

from src.adapters.agent06_verification_handoff import build_agent07_input_from_committed_agent06  # noqa: E402
from src.adapters.claim_verification_context import fingerprint_text  # noqa: E402
from src.adapters.verification_incremental_retriever import Agent07ChromaRetriever  # noqa: E402
from src.adapters.verification_notebook import commit_executed_agent07, execute_prepared_agent07, prepare_agent07_execution  # noqa: E402
from src.adapters.verification_runtime import (  # noqa: E402
    Agent07RuntimeInput,
    VerificationRuntimeDependencies,
    _productive_reverification_input,
)
from src.agents.draft_writing_agent import DraftWritingAgent  # noqa: E402
from src.agents.verification_agent import VerificationAgent  # noqa: E402
from src.config.verification_policy_config import get_verification_input_policy  # noqa: E402
from src.tools.verification.corrections import propose_correction  # noqa: E402
from src.tools.verification.cycle_round_persistence import read_round_status, round_directory  # noqa: E402
from src.tools.verification.resolution import resolve_multiple_correction_proposals  # noqa: E402
from src.tools.verification.validation import build_provisional_verification_traceability_bundle  # noqa: E402

import test_qualitative_correction_keyerror_and_return as R  # noqa: E402

RESULTS = []


def scenario(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
            except Exception:
                RESULTS.append((name, False, traceback.format_exc()))
            else:
                RESULTS.append((name, True, ""))

        return wrapper

    return decorator


CLAIM_TEXT_WRONG = "El método Alpha utiliza una red neuronal convolucional para clasificar documentos científicos."
CLAIM_TEXT_EVIDENCE = "El método Alpha utiliza una arquitectura basada en transformadores para clasificar documentos científicos."
REAL_CLAIM_UID = "11111111-1111-1111-1111-111111111111"


def _run_verification_with_uid(*, store, project_dir, mapping_path, claim_uid):
    """Reutiliza R._run_verification (el flujo real y completo ya
    probado en R05/R06 -- correction_context_factory, proposal_runner,
    reverification_input_factory, correction_llm, reverification_llm,
    todos reales) -- la ÚNICA diferencia es que el contexto del claim
    lleva claim_uid, vía un parche temporal sobre R._clean_claim_context
    que delega en la función real y solo agrega el campo nuevo."""

    real_clean_claim_context = R._clean_claim_context

    def _clean_claim_context_with_uid(**kwargs):
        ctx = real_clean_claim_context(**kwargs)
        return {**ctx, "claim_uid": claim_uid}

    with patch.object(R, "_clean_claim_context", side_effect=_clean_claim_context_with_uid):
        return R._run_verification(
            store=store, project_dir=project_dir, mapping_path=mapping_path,
            claim_text=CLAIM_TEXT_WRONG, evidence_text=CLAIM_TEXT_EVIDENCE,
        )


@scenario("P01. Integración real: claim_uid minteado en 06 sobrevive intacto una verificación real de 07 (RETURN) y una revisión real de 06, incluso corrigiendo el texto del claim")
def test_claim_uid_survives_real_return_revision_cycle():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = T._seed_project(Path(tmp))

        # --- 1. 07 real, con el contexto del claim llevando claim_uid real (como lo haría un handoff real de 06) ---
        executed_first = _run_verification_with_uid(store=store, project_dir=project_dir, mapping_path=mapping_path, claim_uid=REAL_CLAIM_UID)
        assert executed_first.agent_result.requested_transition.action.value == "RETURN"
        commit_executed_agent07(store=store, executed=executed_first)

        cycle_root = Path(tmp)
        revision_request = json.loads(
            (round_directory(cycle_root, "exp_stagespec", 1) / "writer_revision_request.json").read_text()
        )
        # La aserción central de la propagación: el issue real trae el
        # MISMO claim_uid que se declaró en el contexto de 07.
        issue = revision_request["issues"][0]
        assert issue["claim_uid"] == REAL_CLAIM_UID

        # --- 2. 06 real, modo REVISION: el LLM declara CONTINUE con el uid forzado por el issue ---
        runtime06 = E2E.FakeRuntime(outputs=[{
            "section_id": "S1", "section_title": "Modelos predictivos",
            "draft_text": CLAIM_TEXT_EVIDENCE + " [paper_a.pdf | a_chroma].",
            "claims": [{
                "claim": CLAIM_TEXT_EVIDENCE, "supporting_citations": ["[paper_a.pdf | a_chroma]"],
                "identity_action": "CONTINUE", "parent_claim_uids": [REAL_CLAIM_UID],
            }],
        }])
        agent06 = DraftWritingAgent(runtime06)
        output_dir = Path(tmp) / "draft_output"
        previous_draft = {"sections": [
            {"section_id": "S1", "section_title": "Modelos predictivos",
             "draft_text": CLAIM_TEXT_WRONG + " [paper_a.pdf | a_chroma].",
             "claims": [{
                 "claim": CLAIM_TEXT_WRONG, "supporting_citations": ["[paper_a.pdf | a_chroma]"],
                 "claim_uid": REAL_CLAIM_UID, "claim_version": 1, "claim_id": "S1_C1",
                 "parent_claim_uids": [], "claim_text_fingerprint": fingerprint_text(CLAIM_TEXT_WRONG),
                 "created_round": 0, "updated_round": 0,
             }]},
        ]}
        agent_input = E2E._make_agent_input(output_dir, {
            "mode": "REVISION", "writer_revision_request": revision_request, "previous_draft": previous_draft,
            "round_number": 1, "cycle_project_dir": str(cycle_root), "experiment_id": "exp_stagespec",
        })
        with patch("src.agents.draft_writing_agent.validate_draft_dependencies", return_value=E2E._bundle()), \
             patch("src.agents.draft_writing_agent.retrieve_section_evidence",
                   return_value=[{"source_filename": "paper_a.pdf", "chunk_id": "a_chroma", "text": CLAIM_TEXT_EVIDENCE}]):
            result06 = agent06.execute(agent_input)
        assert result06.execution_status.value == "COMPLETED"

        status_after_06 = read_round_status(project_dir=cycle_root, experiment_id="exp_stagespec", round_number=1)
        assert status_after_06["status"] == "REVISION_COMPLETED"

        revised = json.loads((round_directory(cycle_root, "exp_stagespec", 1) / "revised_draft.json").read_text())
        s1 = next(s for s in revised["sections"] if s["section_id"] == "S1")
        assert len(s1["claims"]) == 1
        corrected_claim = s1["claims"][0]

        # La aserción central del test: el claim_uid sobrevivió intacto,
        # el texto SÍ cambió (evidencia de que fue una corrección real,
        # no un no-op), y la versión avanzó.
        assert corrected_claim["claim_uid"] == REAL_CLAIM_UID
        assert corrected_claim["claim"] != CLAIM_TEXT_WRONG
        assert "arquitectura basada en transformadores" in corrected_claim["claim"]
        assert corrected_claim["claim_version"] == 2
        assert corrected_claim["parent_claim_uids"] == [REAL_CLAIM_UID]
        assert corrected_claim["updated_round"] == 1
        assert corrected_claim["created_round"] == 0  # se preserva del padre, no se reinicia


if __name__ == "__main__":
    for fn in (test_claim_uid_survives_real_return_revision_cycle,):
        fn()

    failed = 0
    for name, ok, err in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        if not ok:
            failed += 1
            print(err)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} escenarios OK")
    raise SystemExit(1 if failed else 0)
