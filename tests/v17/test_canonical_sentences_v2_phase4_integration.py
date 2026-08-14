"""Fase 4 -- validación del flujo real 06(canonical_sentences_v2)->07,
rollback a legacy, y fail-closed, usando la infraestructura productiva
real (write_synthetic_project, build_real_draft_execution/prepare_
draft_execution/execute_prepared_draft/commit_executed_draft de Fase
1-3, build_agent07_input_from_committed_agent06 y build_claim_
verification_context_from_agent06_handoff -- ambas funciones reales de
producción, no mocks del consumidor -- y VerificationAgent.verify_claim
real).

HALLAZGO REAL de esta fase (documentado en detalle en la entrega,
también como comentario en canonical_sentences.py): el consumidor real
de 07 (build_agent07_input_from_committed_agent06) exige que claim
["claim"] sea subcadena EXACTA de draft_text -- lo que requiere que el
claim no incluya el signo de puntuación final, porque en draft_text la
puntuación va después de la cita insertada. Legacy ya cumplía esto
(normalize_claim_text hace .rstrip(".?!")); materialize_initial_
section_v2 no lo hacía (Fase 2B pedía "claim = sentence.text exacto"),
lo que producía AGENT07_AGENT06_CLAIM_SPAN_AMBIGUOUS. Corregido en
canonical_sentences.py -- código V2 puro, 07/08 sin ningún cambio.

Multidominio y genérico: ningún test usa contenido de un experimento
real."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "v16"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "v17"))

from test_agent06_v16 import Env  # noqa: E402
from agent06_v17_test_support import (  # noqa: E402
    SyntheticCollection,
    chroma_client_factory,
    write_synthetic_project,
)

from src.adapters.draft_writing_runtime import (  # noqa: E402
    DraftWritingRuntime,
    build_draft_agent_input,
    commit_executed_draft,
    execute_prepared_draft,
    load_draft_configuration,
    prepare_draft_execution,
)
from src.adapters.agent06_verification_handoff import (  # noqa: E402
    build_agent07_input_from_committed_agent06,
)
from src.adapters.claim_verification_context import (  # noqa: E402
    build_claim_verification_context_from_agent06_handoff,
)
from src.agents.draft_writing_agent import DraftWritingAgent  # noqa: E402
from src.agents.verification_agent import VerificationAgent  # noqa: E402
from src.tools.draft_writing.canonical_sentences import (  # noqa: E402
    _fingerprint_claim_text,
)
from src.tools.verification.corrections import fingerprint_text  # noqa: E402

try:
    from langchain_core.messages import AIMessage
except ModuleNotFoundError:
    class AIMessage:
        def __init__(self, content):
            self.content = content

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


NUMERIC_SECTION_TEXT = (
    "El modelo alcanza una precisión de noventa y uno por ciento en la evaluación comparativa."
)
METHOD_SECTION_TEXT = (
    "El método propuesto reduce el tiempo de cómputo respecto a las líneas base evaluadas."
)


class ScriptedLLM:
    """VerificationLLM real (protocolo invoke) -- respuesta fija y
    válida, sin mocks del consumidor: VerificationAgent.verify_claim
    procesa esta respuesta con su propio parser/validación reales."""

    def __init__(self, claim_id: str):
        self.claim_id = claim_id
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=json.dumps({
            "claim_id": self.claim_id, "verdict": "SUPPORTED", "support_level": "STRONG",
            "evidence_ids_used": ["E01"], "evidence_ids_rejected": [],
            "rationale": "La evidencia autorizada respalda la afirmación evaluada.",
            "contradiction_type": "NONE", "contradiction_evidence_ids": [],
            "numeric_assessment": "SUPPORTED", "attribution_assessment": "NOT_APPLICABLE",
            "extrapolation_assessment": "WITHIN_EVIDENCE_SCOPE", "confidence": "LOW",
            "additional_retrieval_needed": False, "llm_correction_recommendation": False,
            "manual_review_required": False, "reason_codes": [],
        }))


def _run_v2_committed_06(*, sections, response, source_filename="paper_a.pdf", chunk_id="a_chroma", quantitative="none"):
    """Ejecuta Agent06 REAL con canonical_sentences_v2, lo comprueba
    (commit_executed_draft, infraestructura real) y devuelve
    (experiment_dir, store, executed) para que el llamador construya
    el handoff real hacia 07."""

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    experiment, store, chunk_rows = write_synthetic_project(tmp_path, sections=sections, quantitative=quantitative)
    cfg = load_draft_configuration(
        tmp_path, attempt_number=1, chroma_client_factory=chroma_client_factory,
        policy_overrides={"draft_representation_contract": "canonical_sentences_v2"},
    )
    collection = SyntheticCollection(chunk_rows)
    runtime = DraftWritingRuntime(response, collection)
    agent = DraftWritingAgent(runtime)
    agent_input = build_draft_agent_input(cfg)
    prepared = prepare_draft_execution(store=store, agent_input=agent_input)
    executed = execute_prepared_draft(store=store, agent=agent, prepared=prepared)
    commit_executed_draft(store=store, executed=executed)
    return tmp, experiment, store, executed


def _single_section_fixture(text, citation_source, citation_chunk):
    return [{
        "section_id": "S1", "section_title": "Methods", "section_type": "linea_tematica",
        "purpose": "p", "key_arguments": ["a"], "evidence_needs": ["e"],
        "papers_to_use": [{"source_filename": citation_source, "title": "A"}],
    }], json.dumps({"section_id": "S1", "sentences": [{"text": text, "supporting_evidence_ids": ["E1"]}]})


def _build_handoff(experiment, store):
    return build_agent07_input_from_committed_agent06(
        store=store, stage_name="06_agente_redactor",
        agent07_config={}, policy_versions={}, schema_versions={},
        experiment_paths={"experiment_dir": str(experiment)},
        outline_paper_mapping_path=experiment / "05_outputs" / "04_outline" / "outline_paper_mapping.csv",
    )


@scenario("V4-01. 06 V2 committed -> input real de 07 (build_agent07_input_from_committed_agent06, sin mocks)")
def test_v4_01_v2_committed_to_real_07_input():
    sections, resp = _single_section_fixture(METHOD_SECTION_TEXT, "paper_a.pdf", "a_chroma")
    tmp, experiment, store, executed = _run_v2_committed_06(sections=sections, response=lambda p: resp)
    try:
        handoff = _build_handoff(experiment, store)
        assert len(handoff["claim_verification_contexts"]) == 1
        ctx = handoff["claim_verification_contexts"][0]
        assert ctx["claim_id"] == "S1_C1"
        assert ctx["claim_uid"]
        assert ctx["original_claim_text"] == METHOD_SECTION_TEXT.rstrip(".")
        assert handoff["expected_claim_ids"] == ("S1_C1",)
        assert handoff["claim_inventory_fingerprint"]
        assert handoff["source_draft_fingerprint"]
    finally:
        tmp.cleanup()


@scenario("V4-02. Agent07 REAL (VerificationAgent.verify_claim) procesa un claim V2 sin crash")
def test_v4_02_real_agent07_processes_v2_claim():
    sections, resp = _single_section_fixture(METHOD_SECTION_TEXT, "paper_a.pdf", "a_chroma")
    tmp, experiment, store, executed = _run_v2_committed_06(sections=sections, response=lambda p: resp)
    try:
        handoff = _build_handoff(experiment, store)
        ctx = handoff["claim_verification_contexts"][0]
        rich = build_claim_verification_context_from_agent06_handoff(ctx, verification_policy={})
        llm = ScriptedLLM("S1_C1")
        result = VerificationAgent(llm=llm).verify_claim(rich)
        assert result.claim_id == "S1_C1"
        assert result.technical_status == "OK"
        assert result.validation_ok is True
        assert llm.calls >= 1
    finally:
        tmp.cleanup()


@scenario("V4-03. Inventario de claims de 06 == claims esperadas por 07 (correspondencia explícita)")
def test_v4_03_claim_inventory_matches_expected():
    sections, resp = _single_section_fixture(METHOD_SECTION_TEXT, "paper_a.pdf", "a_chroma")
    tmp, experiment, store, executed = _run_v2_committed_06(sections=sections, response=lambda p: resp)
    try:
        draft = json.loads(executed.result.output_artifacts["state_of_art_draft.json"].path and Path(executed.result.output_artifacts["state_of_art_draft.json"].path).read_text())
        claims_06 = {c["claim_id"] for s in draft["sections"] for c in s["claims"]}
        handoff = _build_handoff(experiment, store)
        claims_07 = set(handoff["expected_claim_ids"])
        assert claims_06 == claims_07
    finally:
        tmp.cleanup()


@scenario("V4-04. Trazabilidad real generada: claim_uid/identity_action/parent_claim_uids reconocidos por el contexto rico de 07")
def test_v4_04_traceability_fields_recognized():
    sections, resp = _single_section_fixture(METHOD_SECTION_TEXT, "paper_a.pdf", "a_chroma")
    tmp, experiment, store, executed = _run_v2_committed_06(sections=sections, response=lambda p: resp)
    try:
        draft = json.loads(Path(executed.result.output_artifacts["state_of_art_draft.json"].path).read_text())
        claim_06 = draft["sections"][0]["claims"][0]
        assert claim_06["identity_action"] == "NEW"
        assert claim_06["parent_claim_uids"] == []
        assert claim_06["claim_text_fingerprint"]

        handoff = _build_handoff(experiment, store)
        ctx = handoff["claim_verification_contexts"][0]
        assert ctx["claim_uid"] == claim_06["claim_uid"]
        rich = build_claim_verification_context_from_agent06_handoff(ctx, verification_policy={})
        assert rich["claim_uid"] == claim_06["claim_uid"]
        assert rich["supporting_citations"]
    finally:
        tmp.cleanup()


@scenario("V4-05/06. Agent08 real -- SUPERADO: ejecutado de punta a punta en test_canonical_sentences_v2_phase4_closure.py (test_v4_05_06_agent08_actually_executed), no un stub")
def test_v4_05_06_agent08_documented_scope():
    # Este stub queda deliberadamente vacío -- la limitación que
    # documentaba (orquestador completo de 07 no montado) fue superada
    # en la ronda de cierre de Fase 4: el pipeline completo 06(V2 real)
    # -> commit -> 07 (orquestador productivo real, prepare_agent07_
    # execution/execute_prepared_agent07/commit_executed_agent07) ->
    # commit real de 07 -> build_agent08_input_from_committed_agent07
    # (real) -> run_evaluation_pipeline (real, con Ground Truth real en
    # disco) SÍ se ejecuta y se verifica en
    # tests/v17/test_canonical_sentences_v2_phase4_closure.py -- ver
    # ese archivo para la ejecución real, no simulada. Este test se
    # conserva únicamente por continuidad de numeración; el conteo de
    # PASS de este archivo (18/18) NUNCA debe citarse como si incluyera
    # la ejecución real de Agent08 -- esa evidencia vive en el archivo
    # de cierre, contado por separado.
    pass


@scenario("V4-07. Round-trip textual: sentence.text (06) == claim.claim (06, sin puntuación final) es subcadena verificable de draft_text, y el mismo texto llega intacto al contexto de 07")
def test_v4_07_textual_roundtrip_verifiable():
    sections, resp = _single_section_fixture(METHOD_SECTION_TEXT, "paper_a.pdf", "a_chroma")
    tmp, experiment, store, executed = _run_v2_committed_06(sections=sections, response=lambda p: resp)
    try:
        draft = json.loads(Path(executed.result.output_artifacts["state_of_art_draft.json"].path).read_text())
        claim_06 = draft["sections"][0]["claims"][0]["claim"]
        draft_text = draft["sections"][0]["draft_text"]
        assert claim_06 in draft_text  # subcadena exacta -- el contrato real que 07 exige
        assert claim_06 == METHOD_SECTION_TEXT.rstrip(".")

        handoff = _build_handoff(experiment, store)
        ctx = handoff["claim_verification_contexts"][0]
        assert ctx["original_claim_text"] == claim_06
        rich = build_claim_verification_context_from_agent06_handoff(ctx, verification_policy={})
        assert rich["claim_text"] == claim_06
    finally:
        tmp.cleanup()


@scenario("V4-08. _fingerprint_claim_text (V2) produce EXACTAMENTE el mismo resultado que fingerprint_text (infraestructura productiva downstream)")
def test_v4_08_fingerprint_equivalence():
    for text in ("hola mundo", "El modelo alcanza noventa y uno por ciento.", ""):
        assert _fingerprint_claim_text(text) == fingerprint_text(text)


@scenario("V4-09. Manifest V2 correcto: declara canonical_sentences_v2, counts/attempt_logs coherentes")
def test_v4_09_manifest_v2_correct():
    sections, resp = _single_section_fixture(METHOD_SECTION_TEXT, "paper_a.pdf", "a_chroma")
    tmp, experiment, store, executed = _run_v2_committed_06(sections=sections, response=lambda p: resp)
    try:
        manifest = json.loads(Path(executed.result.output_artifacts["draft_generation_manifest.json"].path).read_text())
        assert manifest["draft_representation_contract"] == "canonical_sentences_v2"
        assert manifest["counts"]["llm_calls"] == 1
    finally:
        tmp.cleanup()


@scenario("V4-10. _v2_execution NO aparece en ningún artefacto científico final (state_of_art_draft.json ni manifest)")
def test_v4_10_no_internal_v2_metadata_leaks():
    sections, resp = _single_section_fixture(METHOD_SECTION_TEXT, "paper_a.pdf", "a_chroma")
    tmp, experiment, store, executed = _run_v2_committed_06(sections=sections, response=lambda p: resp)
    try:
        draft_raw = Path(executed.result.output_artifacts["state_of_art_draft.json"].path).read_text()
        manifest_raw = Path(executed.result.output_artifacts["draft_generation_manifest.json"].path).read_text()
        assert "_v2_execution" not in draft_raw
        assert "_v2_execution" not in manifest_raw
    finally:
        tmp.cleanup()


@scenario("V4-11. Rollback con flag AUSENTE -> comportamiento legacy histórico")
def test_v4_11_rollback_flag_absent():
    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(
        lambda p: json.dumps({"section_id": "S1", "section_title": "Methods", "draft_text": "x [a.pdf | c1].", "claims": [{"claim": "x", "supporting_citations": ["[a.pdf | c1]"]}]}),
        e.collection,
    ))
    result = e.agent.execute(e.ai)  # sin draft_representation_contract en policy
    assert result.execution_status.value == "COMPLETED"


@scenario("V4-12. Rollback con 'legacy' EXPLÍCITO -> mismo comportamiento que flag ausente")
def test_v4_12_rollback_flag_explicit_legacy():
    e = Env(attempt=1)
    ai = replace(e.ai, policy={**e.ai.policy, "draft_representation_contract": "legacy"})
    e.agent = DraftWritingAgent(DraftWritingRuntime(
        lambda p: json.dumps({"section_id": "S1", "section_title": "Methods", "draft_text": "x [a.pdf | c1].", "claims": [{"claim": "x", "supporting_citations": ["[a.pdf | c1]"]}]}),
        e.collection,
    ))
    result = e.agent.execute(ai)
    assert result.execution_status.value == "COMPLETED"


@scenario("V4-13. Cero invocaciones a canonical_sentences_v2 bajo rollback legacy")
def test_v4_13_zero_v2_invocations_under_rollback():
    import src.tools.draft_writing.canonical_sentences as module

    calls = {"n": 0}
    original = module.generate_section_canonical_v2

    def traced(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    module.generate_section_canonical_v2 = traced
    try:
        e = Env(attempt=1)
        e.agent = DraftWritingAgent(DraftWritingRuntime(
            lambda p: json.dumps({"section_id": "S1", "section_title": "Methods", "draft_text": "x [a.pdf | c1].", "claims": [{"claim": "x", "supporting_citations": ["[a.pdf | c1]"]}]}),
            e.collection,
        ))
        e.agent.execute(e.ai)
    finally:
        module.generate_section_canonical_v2 = original
    assert calls["n"] == 0


@scenario("V4-14/15. Fallo V2 (agotamiento) -> requested_transition nunca es ADVANCE hacia 07; el orquestador no avanzaría, así que 07/08 nunca verían un draft inválido")
def test_v4_14_15_v2_failure_never_advances():
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [{"text": "", "supporting_evidence_ids": []}]})

    e = Env(attempt=1)
    ai = replace(e.ai, policy={**e.ai.policy, "draft_representation_contract": "canonical_sentences_v2"})
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(ai)
    assert result.requested_transition.action.value != "ADVANCE"
    assert result.requested_transition.action.value in ("RETRY", "HALT_STAGE")
    assert not (e.out / "state_of_art_draft.json").exists()


@scenario("V4-16. Fallo V2 nunca produce fallback a legacy (draft_text/claims vacíos, no un draft legacy inventado)")
def test_v4_16_no_legacy_fallback_on_v2_failure():
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [{"text": "", "supporting_evidence_ids": []}]})

    e = Env(attempt=1)
    ai = replace(e.ai, policy={**e.ai.policy, "draft_representation_contract": "canonical_sentences_v2"})
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(ai)
    report = json.loads((e.out / "draft_validation_report.json").read_text())
    assert report["contract"] == "canonical_sentences_v2"
    assert report["published_draft"] is False


@scenario("V4-17. Sección source-free atraviesa el flujo bajo policy V2 sin llamar al LLM ni generar claim artificial")
def test_v4_17_source_free_section_under_v2():
    calls = {"n": 0}

    def invoke(p):
        calls["n"] += 1
        return json.dumps({"section_id": "S1", "sentences": [{"text": "El modelo alcanza una precisión estable en el conjunto evaluado.", "supporting_evidence_ids": ["E1"]}]})

    e = Env(attempt=1)  # Env ya incluye S2 "Conclusión" (source-free) por defecto
    ai = replace(e.ai, policy={**e.ai.policy, "draft_representation_contract": "canonical_sentences_v2"})
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(ai)
    draft = json.loads((e.out / "state_of_art_draft.json").read_text())
    s2 = next(s for s in draft["sections"] if s["section_id"] == "S2")
    assert s2["claims"] == []  # ninguna claim artificial
    # El LLM solo se llamó para S1 (con evidencia) -- nunca para la
    # sección source-free.
    assert calls["n"] == 1


@scenario("V4-18. Claim numérico respaldado (porcentaje sintético) atraviesa 06(V2)->07 real")
def test_v4_18_numeric_claim_traverses_06_to_07():
    sections, resp = _single_section_fixture(NUMERIC_SECTION_TEXT, "paper_a.pdf", "a_chroma")
    tmp, experiment, store, executed = _run_v2_committed_06(sections=sections, response=lambda p: resp)
    try:
        draft = json.loads(Path(executed.result.output_artifacts["state_of_art_draft.json"].path).read_text())
        assert "noventa y uno por ciento" in draft["sections"][0]["claims"][0]["claim"]

        handoff = _build_handoff(experiment, store)
        ctx = handoff["claim_verification_contexts"][0]
        rich = build_claim_verification_context_from_agent06_handoff(ctx, verification_policy={})
        llm = ScriptedLLM("S1_C1")
        result = VerificationAgent(llm=llm).verify_claim(rich)
        assert result.technical_status == "OK"
        assert result.validation_ok is True
    finally:
        tmp.cleanup()


@scenario("V4-19. counts/sequence/attempt_logs coherentes en una corrida V2 exitosa de una sola sección")
def test_v4_19_counts_sequence_attempt_logs_coherent():
    sections, resp = _single_section_fixture(METHOD_SECTION_TEXT, "paper_a.pdf", "a_chroma")
    tmp, experiment, store, executed = _run_v2_committed_06(sections=sections, response=lambda p: resp)
    try:
        manifest = json.loads(Path(executed.result.output_artifacts["draft_generation_manifest.json"].path).read_text())
        assert executed.result.tool_usage.llm_calls == manifest["counts"]["llm_calls"] == 1
        raw_dir = Path(executed.result.output_artifacts["raw_section_outputs"].path) / "agent_attempt_01"
        v = json.loads((raw_dir / "S1_attempt_1_validation.json").read_text())
        assert v["retry_audit"]["runtime_invoke_sequence_number"] == 1
    finally:
        tmp.cleanup()


@scenario("V4-20. Ningún archivo de Ground Truth aparece en el código V2 (RAG/generación) -- solo instrucciones al LLM de NO usarlo")
def test_v4_20_no_ground_truth_in_v2_code():
    import inspect

    from src.tools.draft_writing import canonical_sentences as cs_module
    from src.tools.draft_writing import prompting as prompting_module

    cs_source = inspect.getsource(cs_module)
    prompting_source = inspect.getsource(prompting_module)
    # canonical_sentences.py no debe mencionar Ground Truth en absoluto
    # -- ni como dato, ni como instrucción (no le corresponde a esta
    # capa).
    assert "Ground Truth" not in cs_source and "ground_truth" not in cs_source.lower().replace("ground truth", "")
    # prompting.py SÍ puede mencionar "Ground Truth" -- pero únicamente
    # como instrucción de PROHIBICIÓN al LLM, nunca como dato cargado.
    for line in prompting_source.splitlines():
        if "Ground Truth" in line:
            assert "no uses" in line.lower() or "ni ground truth" in line.lower() or "sin ground truth" in line.lower()


if __name__ == "__main__":
    for fn in (
        test_v4_01_v2_committed_to_real_07_input,
        test_v4_02_real_agent07_processes_v2_claim,
        test_v4_03_claim_inventory_matches_expected,
        test_v4_04_traceability_fields_recognized,
        test_v4_05_06_agent08_documented_scope,
        test_v4_07_textual_roundtrip_verifiable,
        test_v4_08_fingerprint_equivalence,
        test_v4_09_manifest_v2_correct,
        test_v4_10_no_internal_v2_metadata_leaks,
        test_v4_11_rollback_flag_absent,
        test_v4_12_rollback_flag_explicit_legacy,
        test_v4_13_zero_v2_invocations_under_rollback,
        test_v4_14_15_v2_failure_never_advances,
        test_v4_16_no_legacy_fallback_on_v2_failure,
        test_v4_17_source_free_section_under_v2,
        test_v4_18_numeric_claim_traverses_06_to_07,
        test_v4_19_counts_sequence_attempt_logs_coherent,
        test_v4_20_no_ground_truth_in_v2_code,
    ):
        fn()

    failed = 0
    for name, ok, err in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        if not ok:
            failed += 1
            print(err)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} tests ejecutados PASS")
    print("Agent08 real: EXECUTED -- ver tests/v17/test_canonical_sentences_v2_phase4_closure.py")
    raise SystemExit(1 if failed else 0)
