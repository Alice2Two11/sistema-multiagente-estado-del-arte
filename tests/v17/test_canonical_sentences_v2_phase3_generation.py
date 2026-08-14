"""Fase 3 -- conexión REAL de canonical_sentences_v2 al runtime de
Agent06, exclusivamente detrás de draft_representation_contract ==
"canonical_sentences_v2". Legacy permanece intacto (ver LEGACY 11/11,
tests/v17/test_canonical_sentences_v2_legacy_isolation.py).

CONTRATO EVIDENCE HANDLES: el LLM referencia evidencia mediante
identificadores opacos ("E1", "E2", ...) asignados determinísticamente
por el sistema -- nunca escribe source_filename/chunk_id/
supporting_citations directamente. En el fixture estándar de Env
(tests/v16/test_agent06_v16.py) solo hay un chunk de evidencia
disponible (a.pdf|c1) para S1, así que ese chunk siempre corresponde a
"E1".

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

from test_agent06_v16 import Env  # noqa: E402

from src.adapters.draft_writing_runtime import DraftWritingRuntime  # noqa: E402
from src.agents.draft_writing_agent import DraftWritingAgent  # noqa: E402
from src.tools.draft_writing.canonical_sentences import (  # noqa: E402
    CanonicalSectionValidationFailedV2,
    generate_section_canonical_v2,
)
from src.tools.draft_writing.prompting import build_section_prompt_v2  # noqa: E402

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


def _v2_ai(env, extra_policy=None):
    policy = {**env.ai.policy, "draft_representation_contract": "canonical_sentences_v2"}
    if extra_policy:
        policy.update(extra_policy)
    return replace(env.ai, policy=policy)


def _ok_response(section_id="S1", text="El modelo alcanza una precisión estable en el conjunto de prueba evaluado.", evidence_id="E1"):
    return json.dumps({"section_id": section_id, "sentences": [{"text": text, "supporting_evidence_ids": [evidence_id] if evidence_id else []}]})


@scenario("V3-01. Flag legacy no invoca nada V2 (ausencia del flag)")
def test_legacy_flag_never_invokes_v2():
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


@scenario("V3-02. Flag V2 sí invoca el runtime V2 (build_section_prompt_v2, no el legacy)")
def test_v2_flag_invokes_v2_runtime():
    seen_prompts = []

    def invoke(prompt):
        seen_prompts.append(prompt)
        return _ok_response()

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert len(seen_prompts) >= 1
    assert result.execution_status.value == "COMPLETED"


@scenario("V3-03. El prompt V2 no pide draft_text ni claims -- pide sentences[] con supporting_evidence_ids, nunca supporting_citations/source_filename/chunk_id")
def test_v2_prompt_does_not_ask_for_draft_text_or_claims():
    section = {"section_id": "S1", "section_title": "Methods"}
    prompt = build_section_prompt_v2(section, [], {}, [], {})
    assert '"sentences"' in prompt
    assert '"supporting_evidence_ids"' in prompt
    assert '"draft_text": ""' not in prompt
    assert '"claims": [' not in prompt
    assert '"supporting_citations"' not in prompt.split("FORMATO EXACTO")[1].split("SECCIÓN DEL ESQUEMA")[0]


@scenario("V3-04. Respuesta V2 válida (E1) -> materialización correcta (claim == sentence.text sin puntuación)")
def test_valid_v2_response_materializes_correctly():
    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(lambda p: _ok_response(), e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert result.quality_status.value == "APPROVED"
    draft = json.loads((e.out / "state_of_art_draft.json").read_text())
    s1 = next(s for s in draft["sections"] if s["section_id"] == "S1")
    assert s1["claims"][0]["claim"] == "El modelo alcanza una precisión estable en el conjunto de prueba evaluado"
    assert s1["claims"][0]["supporting_citations"] == ["[a.pdf | c1]"]


@scenario("V3-05. Dos oraciones en un item -> retry")
def test_two_sentences_in_one_item_triggers_retry():
    responses = [
        json.dumps({"section_id": "S1", "sentences": [{"text": "Primera oración. Segunda oración en el mismo item.", "supporting_evidence_ids": ["E1"]}]}),
        _ok_response(),
    ]
    calls = {"n": 0}

    def invoke(p):
        r = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return r

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert calls["n"] == 2
    assert result.quality_status.value == "APPROVED"


@scenario("V3-06. Handle de evidencia inexistente (E99) -> retry, INVALID_EVIDENCE_ID")
def test_invalid_citation_triggers_retry():
    responses = [
        json.dumps({"section_id": "S1", "sentences": [{"text": "El modelo alcanza una precisión estable en el conjunto de prueba evaluado.", "supporting_evidence_ids": ["E99"]}]}),
        _ok_response(),
    ]
    calls = {"n": 0}

    def invoke(p):
        r = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return r

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert calls["n"] == 2
    assert result.quality_status.value == "APPROVED"


@scenario("V3-07. Cita inline en text -> retry")
def test_inline_citation_triggers_retry():
    responses = [
        json.dumps({"section_id": "S1", "sentences": [{"text": "El modelo alcanza [a.pdf | c1] una precisión estable en el conjunto evaluado.", "supporting_evidence_ids": ["E1"]}]}),
        _ok_response(),
    ]
    calls = {"n": 0}

    def invoke(p):
        r = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return r

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert calls["n"] == 2
    assert result.quality_status.value == "APPROVED"


@scenario("V3-08. section_id incorrecto -> retry")
def test_wrong_section_id_triggers_retry():
    responses = [
        _ok_response(section_id="S_WRONG"),
        _ok_response(),
    ]
    calls = {"n": 0}

    def invoke(p):
        r = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return r

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert calls["n"] == 2
    assert result.quality_status.value == "APPROVED"


@scenario("V3-09. El segundo intento recibe los códigos de error del primero en el prompt (INVALID_EVIDENCE_ID)")
def test_second_attempt_receives_first_attempt_errors():
    responses = [
        json.dumps({"section_id": "S1", "sentences": [{"text": "El modelo alcanza una precisión estable en el conjunto de prueba evaluado.", "supporting_evidence_ids": ["E99"]}]}),
        _ok_response(),
    ]
    seen_prompts = []
    calls = {"n": 0}

    def invoke(p):
        seen_prompts.append(p)
        r = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return r

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    e.agent.execute(_v2_ai(e))
    assert len(seen_prompts) == 2
    assert "INVALID_EVIDENCE_ID" in seen_prompts[1]


@scenario("V3-10. Respuestas inválidas en todos los intentos -> nunca fallback legacy (contrato NEEDS_REVISION normal, no FAILED)")
def test_all_attempts_invalid_never_falls_back_to_legacy():
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [{"text": "", "supporting_evidence_ids": []}]})

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert result.execution_status.value == "COMPLETED"
    assert result.quality_status.value == "NEEDS_REVISION"
    assert result.requested_transition.action.value == "RETRY"
    assert not (e.out / "state_of_art_draft.json").exists()
    assert result.decision.code == "SECTION_VALIDATION_FAILED"


@scenario("V3-11. Cero llamadas al normalizador legacy durante una corrida V2")
def test_zero_legacy_normalizer_calls_during_v2_run():
    import src.tools.draft_writing.normalization as norm_module

    calls = {"n": 0}
    original = norm_module.normalize_generated_section

    def traced(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    norm_module.normalize_generated_section = traced
    try:
        e = Env(attempt=1)
        e.agent = DraftWritingAgent(DraftWritingRuntime(lambda p: _ok_response(), e.collection))
        e.agent.execute(_v2_ai(e))
    finally:
        norm_module.normalize_generated_section = original
    assert calls["n"] == 0


@scenario("V3-12. Cero fuzzy matching -- el detector de conectores legacy nunca se invoca en V2")
def test_zero_fuzzy_matching_connector_detector_never_called():
    import src.tools.draft_writing.normalization as norm_module

    calls = {"n": 0}
    original = norm_module.detect_claims_missing_leading_discourse_connector

    def traced(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    norm_module.detect_claims_missing_leading_discourse_connector = traced
    try:
        e = Env(attempt=1)
        e.agent = DraftWritingAgent(DraftWritingRuntime(lambda p: _ok_response(), e.collection))
        e.agent.execute(_v2_ai(e))
    finally:
        norm_module.detect_claims_missing_leading_discourse_connector = original
    assert calls["n"] == 0


@scenario("V3-13. materialize_initial_section_v2 recibe únicamente el payload ya validado (validation_ok=True)")
def test_materializer_receives_only_validated_payload():
    import src.tools.draft_writing.canonical_sentences as module

    received = []
    original = module.materialize_initial_section_v2

    def traced(sentences, section_id, **kwargs):
        received.append(sentences)
        return original(sentences, section_id, **kwargs)

    module.materialize_initial_section_v2 = traced
    try:
        e = Env(attempt=1)
        e.agent = DraftWritingAgent(DraftWritingRuntime(lambda p: _ok_response(), e.collection))
        e.agent.execute(_v2_ai(e))
    finally:
        module.materialize_initial_section_v2 = original
    assert len(received) == 1
    assert received[0][0]["text"] == "El modelo alcanza una precisión estable en el conjunto de prueba evaluado."


@scenario("V3-14. claim resultante sigue siendo copia exacta de sentence.text SIN puntuación final (ver V2B04), de punta a punta")
def test_claim_is_exact_copy_end_to_end():
    text = "El resultado observado replica de forma consistente los hallazgos reportados previamente."
    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(lambda p: _ok_response(text=text), e.collection))
    e.agent.execute(_v2_ai(e))
    draft = json.loads((e.out / "state_of_art_draft.json").read_text())
    s1 = next(s for s in draft["sections"] if s["section_id"] == "S1")
    assert s1["claims"][0]["claim"] == text.rstrip(".")


@scenario("V3-15. El manifest V2 declara explícitamente canonical_sentences_v2")
def test_manifest_declares_v2_contract():
    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(lambda p: _ok_response(), e.collection))
    e.agent.execute(_v2_ai(e))
    manifest = json.loads((e.out / "draft_generation_manifest.json").read_text())
    assert manifest["draft_representation_contract"] == "canonical_sentences_v2"


@scenario("V3-16. El fingerprint V2 difiere del fingerprint legacy para el mismo resto de policy")
def test_v2_fingerprint_differs_from_legacy():
    from src.adapters.draft_writing_runtime import _draft_signature
    from src.state.fingerprints import fingerprint_mapping

    base_policy = {
        "stage_version": "S", "prompt_version": "P", "rag_version": "R", "validation_version": "V",
        "normalization_version": "N",
    }
    cfg_common = {"experiment_id": "e", "experiment_dir": "d", "model": "m", "embedding_model_name": "em", "chroma_collection_name": "c"}
    fp_legacy = fingerprint_mapping(_draft_signature({**cfg_common, "policy": dict(base_policy)}, {}))
    fp_v2 = fingerprint_mapping(_draft_signature({**cfg_common, "policy": {**base_policy, "draft_representation_contract": "canonical_sentences_v2"}}, {}))
    assert fp_legacy != fp_v2


@scenario("V3-17. El fingerprint legacy permanece histórico (idéntico con flag ausente o 'legacy' explícito)")
def test_legacy_fingerprint_remains_historical():
    from src.adapters.draft_writing_runtime import _draft_signature
    from src.state.fingerprints import fingerprint_mapping

    base_policy = {
        "stage_version": "S", "prompt_version": "P", "rag_version": "R", "validation_version": "V",
        "normalization_version": "N",
    }
    cfg_common = {"experiment_id": "e", "experiment_dir": "d", "model": "m", "embedding_model_name": "em", "chroma_collection_name": "c"}
    fp_absent = fingerprint_mapping(_draft_signature({**cfg_common, "policy": dict(base_policy)}, {}))
    fp_explicit_legacy = fingerprint_mapping(_draft_signature({**cfg_common, "policy": {**base_policy, "draft_representation_contract": "legacy"}}, {}))
    assert fp_absent == fp_explicit_legacy


@scenario("V3-18. El output externo de la sección cumple el contrato actual de Agent06 (mismas claves que consume build_draft_reports)")
def test_section_output_matches_current_agent06_contract():
    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(lambda p: _ok_response(), e.collection))
    e.agent.execute(_v2_ai(e))
    draft = json.loads((e.out / "state_of_art_draft.json").read_text())
    s1 = next(s for s in draft["sections"] if s["section_id"] == "S1")
    for key in ("section_id", "section_title", "draft_text", "claims"):
        assert key in s1
    assert isinstance(s1["claims"], list)
    assert isinstance(s1["draft_text"], str)
    assert (e.out / "draft_sections.csv").exists()
    assert (e.out / "draft_claim_evidence.csv").exists()


@scenario("V3-19. Instrumentación R5 presente en cada intento V2")
def test_r5_instrumentation_present_in_each_v2_attempt():
    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(lambda p: _ok_response(), e.collection))
    e.agent.execute(_v2_ai(e))
    attempt_dir = e.out / "raw_section_outputs" / "agent_attempt_01"
    v = json.loads((attempt_dir / "S1_attempt_1_validation.json").read_text())
    audit = v["retry_audit"]
    for key in (
        "runtime_invoke_executed", "runtime_invoke_sequence_number",
        "runtime_response_metadata", "prompt_sha256", "raw_response_sha256",
        "previous_errors_codes_used_in_prompt",
    ):
        assert key in audit, key


@scenario("V3-20. Mismo prompt/input no implica reutilización de raw response entre intentos -- metadata demuestra invocaciones reales distintas")
def test_same_input_does_not_imply_response_reuse_across_attempts():
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [{"text": "", "supporting_evidence_ids": []}]})

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    e.agent.execute(_v2_ai(e))
    attempt_dir = e.out / "raw_section_outputs" / "agent_attempt_01"
    audits = []
    for p in sorted(attempt_dir.glob("S1_attempt_*_validation.json")):
        audits.append(json.loads(p.read_text())["retry_audit"])
    sequence = [a["runtime_invoke_sequence_number"] for a in audits]
    assert sequence == sorted(set(sequence))
    assert all(a["runtime_invoke_executed"] is True for a in audits)


@scenario("V3-21. Sección V2 válida con una llamada -> manifest counts.llm_calls == 1 y ToolUsage.llm_calls == 1")
def test_v3_21_single_call_counters_correct():
    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(lambda p: _ok_response(), e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert result.tool_usage.llm_calls == 1
    manifest = json.loads((e.out / "draft_generation_manifest.json").read_text())
    assert manifest["counts"]["llm_calls"] == 1


@scenario("V3-22. Sección que requiere 2 intentos -> ambos contadores reportan 2")
def test_v3_22_two_attempts_counters_correct():
    responses = [
        json.dumps({"section_id": "S1", "sentences": [{"text": "Primera. Segunda oración en el mismo item.", "supporting_evidence_ids": ["E1"]}]}),
        _ok_response(),
    ]
    calls = {"n": 0}

    def invoke(p):
        r = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return r

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert result.tool_usage.llm_calls == 2
    manifest = json.loads((e.out / "draft_generation_manifest.json").read_text())
    assert manifest["counts"]["llm_calls"] == 2


def _two_section_env_with_evidence():
    e = Env(attempt=1)
    outline_path = e.inp / "outline.json"
    outline = json.loads(outline_path.read_text())
    outline["sections"] = [
        {"section_id": "S1", "section_title": "Methods", "section_type": "linea_tematica", "purpose": "p", "key_arguments": ["a"], "evidence_needs": ["e"], "papers_to_use": [{"source_filename": "a.pdf", "title": "A"}]},
        {"section_id": "S2", "section_title": "Results", "section_type": "linea_tematica", "purpose": "p", "key_arguments": ["a"], "evidence_needs": ["e"], "papers_to_use": [{"source_filename": "a.pdf", "title": "A"}]},
    ]
    outline_path.write_text(json.dumps(outline), encoding="utf-8")
    (e.inp / "mapping.csv").write_text("section_id,source_filename,title\nS1,a.pdf,A\nS2,a.pdf,A\n")
    return e


@scenario("V3-23. Dos secciones V2 con llamadas reales -> secuencia GLOBAL sin reiniciarse (1, 2, ...) -- NO solo monotonía dentro de una sección (eso ya lo cubre V3-20)")
def test_v3_23_global_sequence_across_two_sections():
    e = _two_section_env_with_evidence()
    responses = {
        "S1": _ok_response(section_id="S1", text="El modelo alcanza una precisión estable en el conjunto de prueba evaluado."),
        "S2": _ok_response(section_id="S2", text="El segundo experimento confirma los hallazgos observados en el estudio previo."),
    }

    def invoke(p):
        return responses["S1" if '"S1"' in p else "S2"]

    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    e.agent.execute(_v2_ai(e))
    attempt_dir = e.out / "raw_section_outputs" / "agent_attempt_01"
    seq_s1 = json.loads((attempt_dir / "S1_attempt_1_validation.json").read_text())["retry_audit"]["runtime_invoke_sequence_number"]
    seq_s2 = json.loads((attempt_dir / "S2_attempt_1_validation.json").read_text())["retry_audit"]["runtime_invoke_sequence_number"]
    assert (seq_s1, seq_s2) == (1, 2)


@scenario("V3-24. attempt_logs GLOBAL contiene realmente las validaciones V2 de AMBAS secciones (S1 válida, S2 agota reintentos) -- inspección directa de generation_attempts, no solo archivos")
def test_v3_24_attempt_logs_contain_v2_validations_per_section():
    e = _two_section_env_with_evidence()

    def invoke(p):
        if '"S1"' in p:
            return _ok_response(section_id="S1", text="El modelo alcanza una precisión estable en el conjunto de prueba evaluado.")
        return json.dumps({"section_id": "S2", "sentences": [{"text": "", "supporting_evidence_ids": []}]})

    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert result.quality_status.value == "NEEDS_REVISION"

    report = json.loads((e.out / "draft_validation_report.json").read_text())
    assert report["failed_section"] == "S2"
    generation_attempts = report["generation_attempts"]
    assert set(generation_attempts.keys()) == {"S1", "S2"}
    assert len(generation_attempts["S1"]) == 1
    assert generation_attempts["S1"][0]["contract"] == "canonical_sentences_v2"
    assert generation_attempts["S1"][0]["validation"]["validation_ok"] is True
    assert len(generation_attempts["S2"]) == 3
    for entry in generation_attempts["S2"]:
        assert entry["contract"] == "canonical_sentences_v2"
        assert entry["validation"]["validation_ok"] is False


@scenario("V3-25. Agotamiento en intento externo 1 -> COMPLETED + NEEDS_REVISION + RETRY, nunca FAILED")
def test_v3_25_exhaustion_attempt1_yields_retry():
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [{"text": "", "supporting_evidence_ids": []}]})

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert result.execution_status.value == "COMPLETED"
    assert result.quality_status.value == "NEEDS_REVISION"
    assert result.requested_transition.action.value == "RETRY"


@scenario("V3-26. Agotamiento en intento externo >1 -> COMPLETED + NEEDS_REVISION + HALT_STAGE")
def test_v3_26_exhaustion_attempt2_yields_halt_stage():
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [{"text": "", "supporting_evidence_ids": []}]})

    e = Env(attempt=2)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert result.execution_status.value == "COMPLETED"
    assert result.quality_status.value == "NEEDS_REVISION"
    assert result.requested_transition.action.value == "HALT_STAGE"


@scenario("V3-27. El reporte parcial conserva los códigos V2 EXACTOS del último intento (INVALID_EVIDENCE_ID)")
def test_v3_27_partial_report_preserves_exact_v2_error_codes():
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [{"text": "El modelo alcanza una precisión estable en el conjunto evaluado.", "supporting_evidence_ids": ["E99"]}]})

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    e.agent.execute(_v2_ai(e))
    report = json.loads((e.out / "draft_validation_report.json").read_text())
    assert report["contract"] == "canonical_sentences_v2"
    assert any("INVALID_EVIDENCE_ID" in str(err) for err in report["last_attempt_errors"])


@scenario("V3-28. Sigue sin existir fallback a legacy, incluso tras el cambio de contrato a evidence handles")
def test_v3_28_still_no_legacy_fallback_after_fix():
    import src.tools.draft_writing.normalization as norm_module

    calls = {"n": 0}
    original = norm_module.normalize_generated_section

    def traced(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    norm_module.normalize_generated_section = traced
    try:
        def invoke(p):
            return json.dumps({"section_id": "S1", "sentences": [{"text": "", "supporting_evidence_ids": []}]})

        e = Env(attempt=1)
        e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
        result = e.agent.execute(_v2_ai(e))
    finally:
        norm_module.normalize_generated_section = original
    assert calls["n"] == 0
    assert not (e.out / "state_of_art_draft.json").exists()


@scenario("V3-29. El reporte parcial V2 contiene validation_version")
def test_v3_29_partial_report_contains_validation_version():
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [{"text": "", "supporting_evidence_ids": []}]})

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    e.agent.execute(_v2_ai(e))
    report = json.loads((e.out / "draft_validation_report.json").read_text())
    assert report.get("validation_version") is not None
    assert report["validation_version"] == e.ai.policy.get("validation_version") or isinstance(report["validation_version"], str)


@scenario("V3-30. El reporte parcial V2 contiene section_attempts correcto (número real de intentos agotados)")
def test_v3_30_partial_report_contains_correct_section_attempts():
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [{"text": "", "supporting_evidence_ids": []}]})

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    e.agent.execute(_v2_ai(e))
    report = json.loads((e.out / "draft_validation_report.json").read_text())
    max_attempts = int(e.ai.policy.get("max_section_revision_attempts", 2)) + 1
    assert report["section_attempts"] == max_attempts


@scenario("V3-31. quality_metrics.technical.section_attempts coincide con el reporte")
def test_v3_31_quality_metrics_section_attempts_matches_report():
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [{"text": "", "supporting_evidence_ids": []}]})

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    report = json.loads((e.out / "draft_validation_report.json").read_text())
    assert result.quality_metrics["technical"]["section_attempts"] == report["section_attempts"]


@scenario("V3-32 (evidence handles, regresión genérica). El LLM que intenta devolver supporting_citations directamente -> UNEXPECTED_SENTENCE_FIELD, sin fallback ni reparación, de punta a punta")
def test_v3_32_llm_returning_supporting_citations_rejected_end_to_end():
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [{
            "text": "El modelo alcanza una precisión estable en el conjunto de prueba evaluado.",
            "supporting_citations": ["[a.pdf | c1]"],  # campo prohibido -- el LLM ya no debe producirlo
        }]})

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert result.quality_status.value == "NEEDS_REVISION"
    report = json.loads((e.out / "draft_validation_report.json").read_text())
    assert any("UNEXPECTED_SENTENCE_FIELD" in str(err) and "supporting_citations" in str(err) for err in report["last_attempt_errors"])


if __name__ == "__main__":
    for fn in (
        test_legacy_flag_never_invokes_v2,
        test_v2_flag_invokes_v2_runtime,
        test_v2_prompt_does_not_ask_for_draft_text_or_claims,
        test_valid_v2_response_materializes_correctly,
        test_two_sentences_in_one_item_triggers_retry,
        test_invalid_citation_triggers_retry,
        test_inline_citation_triggers_retry,
        test_wrong_section_id_triggers_retry,
        test_second_attempt_receives_first_attempt_errors,
        test_all_attempts_invalid_never_falls_back_to_legacy,
        test_zero_legacy_normalizer_calls_during_v2_run,
        test_zero_fuzzy_matching_connector_detector_never_called,
        test_materializer_receives_only_validated_payload,
        test_claim_is_exact_copy_end_to_end,
        test_manifest_declares_v2_contract,
        test_v2_fingerprint_differs_from_legacy,
        test_legacy_fingerprint_remains_historical,
        test_section_output_matches_current_agent06_contract,
        test_r5_instrumentation_present_in_each_v2_attempt,
        test_same_input_does_not_imply_response_reuse_across_attempts,
        test_v3_21_single_call_counters_correct,
        test_v3_22_two_attempts_counters_correct,
        test_v3_23_global_sequence_across_two_sections,
        test_v3_24_attempt_logs_contain_v2_validations_per_section,
        test_v3_25_exhaustion_attempt1_yields_retry,
        test_v3_26_exhaustion_attempt2_yields_halt_stage,
        test_v3_27_partial_report_preserves_exact_v2_error_codes,
        test_v3_28_still_no_legacy_fallback_after_fix,
        test_v3_29_partial_report_contains_validation_version,
        test_v3_30_partial_report_contains_correct_section_attempts,
        test_v3_31_quality_metrics_section_attempts_matches_report,
        test_v3_32_llm_returning_supporting_citations_rejected_end_to_end,
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
