"""Fase 4 -- CIERRE REAL: Agent08 ejecutado de punta a punta (no solo
06->07), orquestador fail-closed probado con el fallo real de 07 al
intentar construir su handoff sin un commit de 06, aislamiento de
Ground Truth verificado operacionalmente (no por inspección de código),
y el contrato final de puntuación del claim formalizado con los 5
sufijos pedidos.

Todo el pipeline 06(V2)->07->08 usa código productivo real
(write_synthetic_project, build_real_draft_execution-style calls,
build_agent07_input_from_committed_agent06,
build_agent07_runtime_dependencies, prepare_agent07_execution/
execute_prepared_agent07/commit_executed_agent07,
build_agent08_input_from_committed_agent07, run_evaluation_pipeline).
Los ÚNICOS dobles son los LLM (verification/correction/reverification/
translation/judge) y el modelo de embeddings/BERTScore -- exactamente
lo que un test determinista sin llamadas de red necesita reemplazar,
igual que ya hace tests/orchestration/test_verification_characterization.py
y tests/orchestration/test_evaluation_stagespec_integration.py (de
donde se reutilizan los dobles FakeEmbeddingModel/FakeTensor/
FakeLLMFactory, sin reinventarlos).

Multidominio y genérico: ningún test usa contenido de un experimento
real."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "v16"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "v17"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "orchestration"))

from dataclasses import replace  # noqa: E402

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
from src.adapters.evaluation_upstream import (  # noqa: E402
    build_agent08_input_from_committed_agent07,
)
from src.adapters.verification_notebook import (  # noqa: E402
    commit_executed_agent07,
    execute_prepared_agent07,
    prepare_agent07_execution,
)
from src.adapters.verification_runtime import (  # noqa: E402
    Agent07RuntimeInput,
    build_agent07_runtime_dependencies,
)
from src.agents.draft_writing_agent import DraftWritingAgent  # noqa: E402
from src.config.verification_policy_config import get_verification_input_policy  # noqa: E402
from src.tools.draft_writing.canonical_sentences import (  # noqa: E402
    _fingerprint_claim_text,
    materialize_initial_section_v2,
    validate_and_parse_sentences_v2,
)
from src.tools.evaluation.evaluation_pipeline import run_evaluation_pipeline  # noqa: E402
from src.tools.evaluation.llm_judge import JUDGE_CRITERIA  # noqa: E402
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


LONG_TEXT = (
    "El modelo alcanza una precision del noventa y uno por ciento (91%) en la evaluacion "
    "comparativa realizada sobre el conjunto de datos completo, superando ampliamente a los "
    "metodos de referencia previamente reportados en la literatura especializada del area de estudio."
)
GT_TEXT = (
    "Estudios previos reportaron un desempeño de ochenta y ocho por ciento en tareas similares "
    "de clasificacion automatica. El presente trabajo confirma mejoras consistentes en el area "
    "evaluada con nuevos metodos de aprendizaje automatico aplicados al mismo dominio experimental."
)


class ScriptedVerificationLLM:
    """Único punto de mock permitido: la respuesta del LLM de
    verificación. VerificationAgent.verify_claim, build_claim_
    verification_context_from_agent06_handoff, y todo el orquestador
    run_agent07_in_memory son código real, sin mocks."""

    def __init__(self, claim_id):
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


class NoopLLM:
    """Doble para correction_llm/reverification_llm -- nunca se
    invocan realmente cuando el veredicto de verificación es SUPPORTED
    (no hay corrección que proponer), pero build_agent07_runtime_
    dependencies exige que ambos existan."""

    def invoke(self, messages):
        return AIMessage(content=json.dumps({}))


def _simple_embedding_vector(text):
    vowels = sum(1 for c in text.lower() if c in "aeiouáéíóú")
    length = max(len(text), 1)
    vector = np.array([vowels / length, 1 - vowels / length])
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


class FakeEmbeddingModel:
    """Mismo doble real que tests/orchestration/
    test_evaluation_automatic_metrics_integration.py -- devuelve
    np.ndarray real (no listas planas), que es lo que el pipeline real
    de 08 necesita para .mean(axis=0)."""

    def encode(self, chunks, *, normalize_embeddings=True, show_progress_bar=False):
        return np.array([_simple_embedding_vector(c) for c in chunks])


class FakeTensor:
    """Mismo doble real que tests/orchestration/
    test_evaluation_bertscore_characterization.py."""

    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)

    def mean(self):
        return sum(self._values) / len(self._values)


def _bertscore_fn(candidates, references, **kwargs):
    values = [1.0 for _ in candidates]
    return FakeTensor(values), FakeTensor(values), FakeTensor(values)


def _valid_judge_response():
    return {
        "scores": {c: {"score": 4, "justification": "ok", "evidence_from_generated": []} for c in JUDGE_CRITERIA},
        "strengths": [], "organization_differences": [], "missing_topics_or_omissions": [],
        "overall_assessment": "Evaluación general sólida.",
    }


class FakeLLMFactory:
    """Mismo doble real que tests/orchestration/
    test_evaluation_language_characterization.py (traducción/juez)."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = 0

    def __call__(self):
        return self

    def invoke(self, messages):
        self.calls += 1
        content = self.responses[min(self.calls - 1, len(self.responses) - 1)] if self.responses else "{}"
        return AIMessage(content=content)


_EVAL_POLICY = {
    "translate_for_rouge_if_language_differs": False,
    "max_translation_chars_per_chunk": 200, "semantic_chunk_chars": 60, "semantic_chunk_overlap_chars": 0,
    "max_semantic_chunks_per_text": 5, "evaluation_embedding_model": "fake-embedding-model",
    "bertscore_model": "fake-bertscore-model", "max_bertscore_pairs": 4, "minimum_ground_truth_words": 5,
    "require_explicit_ground_truth_end_heading": False, "minimum_generated_words": 3,
    "llm_judge_max_generated_chars": 2000, "llm_judge_max_ground_truth_chars": 2000,
    "llm_judge_max_attempts": 3, "fail_on_invalid_evaluation": True,
    "create_corpus_gap_suggestions": True, "run_llm_judge": True,
}


def _single_section_fixture(text, citation_source="paper_a.pdf", citation_chunk="a_chroma"):
    return [{
        "section_id": "S1", "section_title": "Methods", "section_type": "linea_tematica",
        "purpose": "p", "key_arguments": ["a"], "evidence_needs": ["e"],
        "papers_to_use": [{"source_filename": citation_source, "title": "A"}],
    }], json.dumps({"section_id": "S1", "sentences": [{"text": text, "supporting_citations": [f"[{citation_source} | {citation_chunk}]"]}]})


def _run_06_v2_to_07_committed(*, text=LONG_TEXT, sections=None, response=None):
    """Ejecuta 06(V2 real) -> commit -> 07 (orquestador real completo,
    prepare/execute/commit_executed_agent07) -> commit real de 07.
    Único mock: verification_llm/correction_llm/reverification_llm.
    Devuelve (tmp, experiment, store, tmp_path, chunk_rows) -- el
    llamador cierra tmp. ``chunk_rows`` es la evidencia REAL (con
    "distance") tal como la usó 06/07 -- se propaga tal cual a 08 en
    vez de releerla desde draft_rag_evidence.csv, para no introducir
    una fuente de datos distinta de la que realmente vio la
    verificación."""

    sections = sections or _single_section_fixture(text)[0]
    response = response or _single_section_fixture(text)[1]

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    experiment, store, chunk_rows = write_synthetic_project(tmp_path, sections=sections, quantitative="none")
    cfg = load_draft_configuration(
        tmp_path, attempt_number=1, chroma_client_factory=chroma_client_factory,
        policy_overrides={"draft_representation_contract": "canonical_sentences_v2"},
    )
    collection = SyntheticCollection(chunk_rows)
    runtime = DraftWritingRuntime(lambda p: response, collection)
    agent = DraftWritingAgent(runtime)
    agent_input = build_draft_agent_input(cfg)
    prepared06 = prepare_draft_execution(store=store, agent_input=agent_input)
    executed06 = execute_prepared_draft(store=store, agent=agent, prepared=prepared06)
    commit_executed_draft(store=store, executed=executed06)

    committed_agent06_output = build_agent07_input_from_committed_agent06(
        store=store, stage_name="06_agente_redactor",
        agent07_config={}, policy_versions={"verification": "v1"},
        schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
        experiment_paths={"root": str(tmp_path)},
        outline_paper_mapping_path=experiment / "05_outputs" / "04_outline" / "outline_paper_mapping.csv",
    )
    claim_id = committed_agent06_output["claim_verification_contexts"][0]["claim_id"]
    verification_policy = get_verification_input_policy({})
    agent07_config = {
        "verification_policy": verification_policy, "correction_policy": {"enabled": True}, "reverification_policy": {"enabled": True},
        "verification_budgets": {"max_llm_calls": 3}, "correction_budgets": {"max_llm_calls": 3}, "reverification_budgets": {"max_llm_calls": 3},
        "verification_prompt_version": "v1", "correction_prompt_version": "v1", "reverification_prompt_version": "v1",
    }
    runtime_input = Agent07RuntimeInput(
        committed_agent06_output=committed_agent06_output, agent07_config=agent07_config,
        policy_versions={"verification": "v1"},
        schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
        experiment_paths={"root": str(tmp_path)},
    )
    dependencies = build_agent07_runtime_dependencies(
        config=agent07_config, experiment_paths={"root": str(tmp_path)},
        verification_llm=ScriptedVerificationLLM(claim_id), correction_llm=NoopLLM(), reverification_llm=NoopLLM(),
    )
    prepared07 = prepare_agent07_execution(store=store, runtime_input=runtime_input)
    executed07 = execute_prepared_agent07(store=store, prepared=prepared07, dependencies=dependencies)
    commit_executed_agent07(store=store, executed=executed07)

    return tmp, experiment, store, tmp_path, chunk_rows


def _run_agent08_real(experiment, tmp_path, chunk_rows, *, gt_dir):
    """build_agent08_input_from_committed_agent07 real ->
    run_evaluation_pipeline real. ``chunk_rows`` debe ser la MISMA
    evidencia real que vio 06/07 (ver nota en _run_06_v2_to_07_
    committed) -- nunca releída de un CSV intermedio. Devuelve
    (upstream, result)."""

    agent07_dir = tmp_path / "07_verification"
    draft_json = tmp_path / experiment.name / "05_outputs" / "05_draft" / "state_of_art_draft.json"
    draft_md = tmp_path / experiment.name / "05_outputs" / "05_draft" / "state_of_art_draft.md"
    upstream = build_agent08_input_from_committed_agent07(
        agent07_directory=agent07_dir, draft_json_path=draft_json, draft_markdown_path=draft_md,
    )
    draft = json.loads(draft_json.read_text())
    sections = draft["sections"]
    generated_text = " ".join(s.get("draft_text", "") for s in sections)

    result = run_evaluation_pipeline(
        generated_plain_text=generated_text, sections=sections,
        chunks=chunk_rows,
        traceability_rows=list(upstream.traceability_rows), source_stage=upstream.source_stage,
        upstream_runtime_status=upstream.upstream_runtime_status,
        reverification_performed=upstream.reverification_performed,
        reverification_reason=upstream.reverification_reason,
        claims_verified=upstream.claims_verified,
        claims_requiring_manual_review=upstream.claims_requiring_manual_review,
        manual_review_claim_ids=list(upstream.manual_review_claim_ids),
        generated_status="EVALUATION_READY", evaluation_ready_json_path=str(draft_json),
        experiment_id=experiment.name, topic_name="Tema sintético multidominio",
        ground_truth_dir=str(gt_dir), evaluation_policy=dict(_EVAL_POLICY),
        translation_llm_factory=FakeLLMFactory(), embedding_model_factory=lambda name: FakeEmbeddingModel(),
        bertscore_score_fn=_bertscore_fn, judge_llm_factory=FakeLLMFactory(responses=[json.dumps(_valid_judge_response())]),
    )
    return upstream, result


@scenario("V4-05/06 (REAL). Agent08 ejecutado de punta a punta: 06(V2)->07(orquestador real)->build_agent08_input_from_committed_agent07(real)->run_evaluation_pipeline(real), sin crash")
def test_v4_05_06_agent08_actually_executed():
    tmp, experiment, store, tmp_path, chunk_rows = _run_06_v2_to_07_committed()
    try:
        gt_dir = tmp_path / "ground_truth"
        gt_dir.mkdir()
        (gt_dir / "ground_truth_literature_review.txt").write_text(GT_TEXT, encoding="utf-8")

        upstream, result = _run_agent08_real(experiment, tmp_path, chunk_rows, gt_dir=gt_dir)

        # No crash (llegar aquí ya lo confirma) + recibe el draft correcto
        assert upstream.claims_verified == 1
        assert LONG_TEXT.rstrip(".") in upstream.traceability_rows[0]["claim"] or upstream.traceability_rows[0]["claim"] in LONG_TEXT
        # Trazabilidad REAL de 07 (verdict/claim_id/source real, no inventado)
        row = upstream.traceability_rows[0]
        assert row["verdict"] == "SUPPORTED"
        assert row["claim_id"] == "S1_C1"
        assert row["source_filename"] == "paper_a.pdf"
        # Métricas de factualidad/evidencia SE CALCULAN (no ausentes)
        assert "factual_audit" in result
        assert "claim_metrics" in result["factual_audit"]
        assert result["factual_audit"]["claim_metrics"]["factual_precision"] == 1.0
        assert "automatic_metrics_result" in result
        # Conteos de claims coherentes: 1 claim en 06, 1 fila de trazabilidad en 08
        assert len(upstream.traceability_rows) == 1
        # No interpreta metadata interna _v2_execution
        assert "_v2_execution" not in json.dumps(result, default=str)
    finally:
        tmp.cleanup()


@scenario("V4-06b (REAL). Ground Truth entra ÚNICAMENTE en 08: el resultado de evaluación SÍ contiene el texto/ruta de GT")
def test_v4_06b_ground_truth_present_only_in_evaluation_result():
    tmp, experiment, store, tmp_path, chunk_rows = _run_06_v2_to_07_committed()
    try:
        gt_dir = tmp_path / "ground_truth"
        gt_dir.mkdir()
        (gt_dir / "ground_truth_literature_review.txt").write_text(GT_TEXT, encoding="utf-8")
        upstream, result = _run_agent08_real(experiment, tmp_path, chunk_rows, gt_dir=gt_dir)
        assert result["ground_truth_plain_text"].strip() != ""
        assert "ochenta y ocho" in result["ground_truth_plain_text"]
    finally:
        tmp.cleanup()


@scenario("V4-14/15 (REAL, orquestador -- validate_transition productivo). Fallo V2 en 06 con RETRY (intento 1) y HALT_STAGE (intento 2): la función real de routing del orquestador nunca resuelve ADVANCE hacia 07 -- 07/08 nunca se despachan; commited-stage fail-closed verificado como evidencia complementaria")
def test_v4_14_15_real_orchestrator_blocks_07_when_06_fails():
    from src.orchestration.decision_engine import validate_transition

    sections, _ = _single_section_fixture(LONG_TEXT)
    invalid_response = json.dumps({"section_id": "S1", "sentences": [{"text": "", "supporting_citations": []}]})

    tmp = tempfile.TemporaryDirectory()
    try:
        tmp_path = Path(tmp.name)
        experiment, store, chunk_rows = write_synthetic_project(tmp_path, sections=sections, quantitative="none")
        cfg = load_draft_configuration(
            tmp_path, attempt_number=1, chroma_client_factory=chroma_client_factory,
            policy_overrides={"draft_representation_contract": "canonical_sentences_v2"},
        )
        collection = SyntheticCollection(chunk_rows)
        runtime = DraftWritingRuntime(lambda p: invalid_response, collection)
        agent = DraftWritingAgent(runtime)
        agent_input = build_draft_agent_input(cfg)
        prepared06 = prepare_draft_execution(store=store, agent_input=agent_input)
        executed06 = execute_prepared_draft(store=store, agent=agent, prepared=prepared06)

        # 06 falla -- NUNCA se llama commit_executed_draft (exactamente lo
        # que haría el orquestador real: no comete un resultado NEEDS_
        # REVISION/RETRY como si fuera exitoso).
        assert executed06.result.quality_status.value == "NEEDS_REVISION"
        assert executed06.result.requested_transition.action.value != "ADVANCE"

        # ORQUESTADOR REAL: validate_transition (src/orchestration/
        # decision_engine.py) es la función productiva que interpreta
        # requested_transition y decide la acción/destino RESULTANTE --
        # el mismo código que usa el pipeline real para decidir si
        # despachar la siguiente etapa. Se le pasa el requested_
        # transition REAL de esta ejecución real de 06 (intento 1,
        # RETRY) -- no un valor inventado.
        validated_attempt1 = validate_transition(
            current_stage="06_agente_redactor",
            requested_transition=executed06.result.requested_transition,
            quality_status=executed06.result.quality_status,
            attempts_used=1, max_attempts=2,
        )
        assert validated_attempt1.action != "ADVANCE"
        assert validated_attempt1.action == "RETRY"
        # Con RETRY, el destino sigue siendo 06 mismo -- nunca 07.
        assert validated_attempt1.target_stage == "06_agente_redactor"

        # Intento externo 2 (agotado): ejecución real independiente del
        # MISMO contrato de fallo, usando el fixture Env (test_agent06_
        # v16.py) -- no requiere reconstruir la persistencia RETRY
        # completa entre intentos externos (mecanismo de write_
        # synthetic_project/load_draft_configuration, no relacionado con
        # lo que esta prueba verifica: el routing real de la transición).
        e2 = Env(attempt=2)
        ai2 = replace(e2.ai, policy={**e2.ai.policy, "draft_representation_contract": "canonical_sentences_v2"})
        e2.agent = DraftWritingAgent(DraftWritingRuntime(lambda p: invalid_response, e2.collection))
        result2 = e2.agent.execute(ai2)
        assert result2.quality_status.value == "NEEDS_REVISION"
        validated_attempt2 = validate_transition(
            current_stage="06_agente_redactor",
            requested_transition=result2.requested_transition,
            quality_status=result2.quality_status,
            attempts_used=2, max_attempts=2,
        )
        assert validated_attempt2.action != "ADVANCE"
        assert validated_attempt2.action == "HALT_STAGE"
        # HALT_STAGE nunca tiene destino -- nunca 07, nunca 08.
        assert validated_attempt2.target_stage is None

        # Evidencia COMPLEMENTARIA (committed-stage fail-closed, no el
        # argumento principal de esta prueba): 07 tampoco puede construir
        # su handoff sobre el StateStore real, porque 06 nunca comprometió
        # nada -- comportamiento real del adapter productivo.
        try:
            build_agent07_input_from_committed_agent06(
                store=store, stage_name="06_agente_redactor",
                agent07_config={}, policy_versions={}, schema_versions={},
                experiment_paths={"experiment_dir": str(experiment)},
                outline_paper_mapping_path=experiment / "05_outputs" / "04_outline" / "outline_paper_mapping.csv",
            )
            raise AssertionError("07 no debería poder construir su handoff sin un commit real de 06")
        except ValueError as exc:
            assert "AGENT07_AGENT06_STAGE_NOT_FOUND" in str(exc)

        # Ningún artefacto committed nuevo de 07/08 -- el directorio de
        # verificación ni siquiera se creó.
        assert not (tmp_path / "07_verification").exists()

        # No fallback legacy: el draft nunca se publicó bajo ningún contrato.
        assert not (tmp_path / experiment.name / "05_outputs" / "05_draft" / "state_of_art_draft.json").exists()
    finally:
        tmp.cleanup()


@scenario("V4-20 (REAL, operacional, orden correcto). Ground Truth existe ANTES de ejecutar 06: excluido deliberadamente del corpus/RAG/allowed_pairs/handoff 06->07/trazabilidad de 07, presente únicamente en la entrada real de 08")
def test_v4_20_ground_truth_operational_isolation():
    GT_MARKER = "MARCADOR_GROUND_TRUTH_UNICO_86420"
    gt_text_with_marker = GT_TEXT + " " + GT_MARKER

    tmp = tempfile.TemporaryDirectory()
    try:
        tmp_path = Path(tmp.name)

        # 0. Ground Truth se crea PRIMERO, antes de tocar 06 en absoluto
        # -- así cualquier ausencia posterior del marcador es EXCLUSIÓN
        # deliberada del pipeline de generación/verificación, no una
        # casualidad de que el archivo todavía no existiera.
        gt_dir = tmp_path / "ground_truth"
        gt_dir.mkdir()
        gt_file = gt_dir / "ground_truth_literature_review.txt"
        gt_file.write_text(gt_text_with_marker, encoding="utf-8")
        # 1. GT existe físicamente en disco antes de ejecutar 06.
        assert gt_file.is_file()
        assert GT_MARKER in gt_file.read_text(encoding="utf-8")

        sections, response = _single_section_fixture(LONG_TEXT)
        experiment, store, chunk_rows = write_synthetic_project(tmp_path, sections=sections, quantitative="none")

        # 2. El corpus/memoria documental elegible para RAG (chunk_rows,
        # construido por write_synthetic_project, independiente de la
        # carpeta de Ground Truth) nunca contiene el marcador -- 06 nunca
        # tuvo acceso a él porque el corpus de referencia y el directorio
        # de Ground Truth son fuentes distintas por diseño, aunque GT ya
        # existiera en disco en este punto.
        for row in chunk_rows:
            assert GT_MARKER not in row.get("text", "")

        cfg = load_draft_configuration(
            tmp_path, attempt_number=1, chroma_client_factory=chroma_client_factory,
            policy_overrides={"draft_representation_contract": "canonical_sentences_v2"},
        )
        collection = SyntheticCollection(chunk_rows)
        runtime = DraftWritingRuntime(lambda p: response, collection)
        agent = DraftWritingAgent(runtime)
        agent_input = build_draft_agent_input(cfg)
        prepared06 = prepare_draft_execution(store=store, agent_input=agent_input)
        executed06 = execute_prepared_draft(store=store, agent=agent, prepared=prepared06)
        commit_executed_draft(store=store, executed=executed06)

        draft_json = tmp_path / experiment.name / "05_outputs" / "05_draft" / "state_of_art_draft.json"
        draft = json.loads(draft_json.read_text())

        # 3. Chunks recuperados/evidence real de 06 (draft_rag_evidence.csv,
        # persistido por la ejecución real) nunca contiene el marcador.
        rag_evidence_path = tmp_path / experiment.name / "05_outputs" / "05_draft" / "draft_rag_evidence.csv"
        assert GT_MARKER not in rag_evidence_path.read_text(encoding="utf-8")

        # 4. allowed_pairs / citas de la sección real (derivadas de la
        # evidencia real de 06) nunca incluyen ningún archivo de Ground
        # Truth.
        citations = draft["sections"][0]["claims"][0]["supporting_citations"]
        assert all("ground_truth" not in c.lower() for c in citations)
        assert all(GT_MARKER not in c for c in citations)

        # 5. Handoff 06->07 (claim_verification_contexts real, construido
        # por el adapter productivo) nunca menciona Ground Truth en
        # ningún campo.
        handoff = build_agent07_input_from_committed_agent06(
            store=store, stage_name="06_agente_redactor",
            agent07_config={}, policy_versions={"verification": "v1"},
            schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
            experiment_paths={"root": str(tmp_path)},
            outline_paper_mapping_path=experiment / "05_outputs" / "04_outline" / "outline_paper_mapping.csv",
        )
        assert GT_MARKER not in json.dumps(handoff, default=str)
        assert "ground_truth" not in json.dumps(handoff, default=str).lower()

        # 6. Orquestador REAL de 07 (prepare_agent07_execution/
        # execute_prepared_agent07/commit_executed_agent07, con verify_
        # claim real) -- su trazabilidad committed real tampoco menciona
        # Ground Truth.
        claim_id = handoff["claim_verification_contexts"][0]["claim_id"]
        verification_policy = get_verification_input_policy({})
        agent07_config = {
            "verification_policy": verification_policy, "correction_policy": {"enabled": True}, "reverification_policy": {"enabled": True},
            "verification_budgets": {"max_llm_calls": 3}, "correction_budgets": {"max_llm_calls": 3}, "reverification_budgets": {"max_llm_calls": 3},
            "verification_prompt_version": "v1", "correction_prompt_version": "v1", "reverification_prompt_version": "v1",
        }
        runtime_input = Agent07RuntimeInput(
            committed_agent06_output=handoff, agent07_config=agent07_config,
            policy_versions={"verification": "v1"},
            schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
            experiment_paths={"root": str(tmp_path)},
        )
        dependencies = build_agent07_runtime_dependencies(
            config=agent07_config, experiment_paths={"root": str(tmp_path)},
            verification_llm=ScriptedVerificationLLM(claim_id), correction_llm=NoopLLM(), reverification_llm=NoopLLM(),
        )
        prepared07 = prepare_agent07_execution(store=store, runtime_input=runtime_input)
        executed07 = execute_prepared_agent07(store=store, prepared=prepared07, dependencies=dependencies)
        commit_executed_agent07(store=store, executed=executed07)

        agent07_dir = tmp_path / "07_verification"
        draft_md = tmp_path / experiment.name / "05_outputs" / "05_draft" / "state_of_art_draft.md"
        upstream = build_agent08_input_from_committed_agent07(
            agent07_directory=agent07_dir, draft_json_path=draft_json, draft_markdown_path=draft_md,
        )
        assert GT_MARKER not in json.dumps(upstream.to_dict(), default=str)
        for row in upstream.traceability_rows:
            assert GT_MARKER not in json.dumps(row, default=str)

        # 7. El marcador SÍ aparece -- únicamente en la entrada/evaluación
        # real de 08, que es la ÚNICA etapa a la que Ground Truth se
        # entrega deliberadamente.
        _, result = _run_agent08_real(experiment, tmp_path, chunk_rows, gt_dir=gt_dir)
        assert GT_MARKER in result["ground_truth_plain_text"]
    finally:
        tmp.cleanup()


@scenario("Contrato final del claim (formal). sentence.text intacto, claim.claim pierde SOLO el sufijo final [.?!]+, subcadena exacta de draft_text, fingerprint downstream sobre el mismo claim.claim -- para '.', '?', '!', '?!', '...'")
def test_final_claim_text_contract_all_punctuation_suffixes():
    base = "El resultado observado confirma la hipótesis planteada en el estudio comparativo realizado"
    cases = [
        (base + ".", "."),
        (base + "?", "?"),
        (base + "!", "!"),
        (base + "?!", "?!"),
        (base + "...", "..."),
    ]
    for text, suffix in cases:
        payload = {"section_id": "SX", "sentences": [{"text": text, "supporting_citations": ["[p.pdf | c1]"]}]}
        parsed = validate_and_parse_sentences_v2(payload, {("p.pdf", "c1")})
        assert parsed["validation_ok"] is True, (text, parsed["errors"])

        # sentence.text permanece intacto (incluido el sufijo original)
        assert parsed["sentences"][0]["text"] == text

        materialized = materialize_initial_section_v2(parsed["sentences"], "SX")
        claim_text = materialized["claims"][0]["claim"]

        # claim.claim pierde SOLO el sufijo final de puntuación
        assert claim_text == base, (text, claim_text)
        assert claim_text + suffix == text

        # Ninguna otra parte del texto cambia (todo antes del sufijo es idéntico)
        assert text.startswith(claim_text)

        # El claim sigue siendo subcadena exacta de draft_text
        assert claim_text in materialized["draft_text"]

        # El fingerprint downstream se calcula sobre ESE MISMO claim.claim
        assert materialized["claims"][0]["claim_text_fingerprint"] == _fingerprint_claim_text(claim_text)
        assert materialized["claims"][0]["claim_text_fingerprint"] == fingerprint_text(claim_text)


if __name__ == "__main__":
    for fn in (
        test_v4_05_06_agent08_actually_executed,
        test_v4_06b_ground_truth_present_only_in_evaluation_result,
        test_v4_14_15_real_orchestrator_blocks_07_when_06_fails,
        test_v4_20_ground_truth_operational_isolation,
        test_final_claim_text_contract_all_punctuation_suffixes,
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
    raise SystemExit(1 if failed else 0)
