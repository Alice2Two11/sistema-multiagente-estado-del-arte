"""NV2 -- verificación de soporte numérico conectada al contrato V2
(canonical_sentences_v2) ANTES de aceptar cada sección, con salvage
determinista nativo cuando aplique. Cierra la inconsistencia
contractual real (confirmada con evidencia de Exp07): V2 nunca
verificaba soporte numérico dentro de su propio bucle de reintentos --
solo `build_draft_reports` lo detectaba, demasiado tarde, sin
oportunidad de retry/salvage a nivel de sección.

V2 NUNCA importa `validate_generated_section` completo (matching
exacto de claim==oración, citation_errors, etc. -- contrato legacy que
no aplica a V2). Solo reutiliza dos funciones puras extraídas de
`validation.py`: `compute_unsupported_numeric_values` y
`build_section_evidence_numeric_tokens` -- misma semántica histórica
exacta, sin acoplar V2 al resto de las reglas legacy.

Multidominio y genérico: ningún test usa contenido de un experimento
real."""

from __future__ import annotations

import json
import sys
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
    v2_numeric_salvage,
    v2_numeric_support_errors,
)
from src.tools.draft_writing.validation import (  # noqa: E402
    build_draft_reports,
    validate_generated_section,
)

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


@scenario("NV2-01. V2 + número respaldado por la evidencia -> sin salvage, sección aceptada, numeric_failure_count global = 0")
def test_nv2_01_supported_number_no_salvage_needed():
    calls = {"n": 0}

    def invoke(p):
        calls["n"] += 1
        return json.dumps({
            "section_id": "S1",
            "sentences": [{"text": "El modelo alcanza una precision de 95 en el conjunto de prueba evaluado.", "supporting_evidence_ids": ["E1"]}],
        })

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert calls["n"] == 1  # una sola llamada LLM -- ningún retry ni salvage se activó
    assert result.quality_status.value == "APPROVED"
    report = json.loads((e.out / "draft_validation_report.json").read_text())
    assert report["numeric_failure_count"] == 0


@scenario("NV2-02. V2 + número NO respaldado -> se detecta ANTES de aceptar la sección (v2_numeric_support_errors real)")
def test_nv2_02_unsupported_number_detected_before_acceptance():
    evidence = [{"source_filename": "a.pdf", "chunk_id": "c1", "text": "El modelo alcanzo 91% en las pruebas realizadas."}]
    sentences = [{"sentence_id": 0, "text": "El modelo alcanza un 77% de precision en el conjunto evaluado.", "supporting_citations": ["[a.pdf | c1]"]}]
    errors = v2_numeric_support_errors(sentences, evidence)
    assert errors == ["UNSUPPORTED_NUMERIC_VALUE:77%"]


@scenario("NV2-03. Número no respaldado + salvage exitoso -> versión salvaged es la aceptada, número ausente del draft/claims finales, numeric_failure_count global = 0")
def test_nv2_03_salvage_success_produces_clean_accepted_section():
    def invoke(p):
        return json.dumps({
            "section_id": "S1",
            "sentences": [
                {"text": "El modelo alcanza un 77% de precision, un valor no reportado por la evidencia disponible.", "supporting_evidence_ids": ["E1"]},
                {"text": "El segundo hallazgo confirma la consistencia general de los resultados obtenidos.", "supporting_evidence_ids": ["E1"]},
            ],
        })

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert result.quality_status.value == "APPROVED"

    draft = json.loads((e.out / "state_of_art_draft.json").read_text())
    s1 = next(s for s in draft["sections"] if s["section_id"] == "S1")
    assert "77%" not in s1["draft_text"]
    assert all("77%" not in c["claim"] for c in s1["claims"])
    assert s1["draft_text"] == "El segundo hallazgo confirma la consistencia general de los resultados obtenidos [a.pdf | c1]."

    report = json.loads((e.out / "draft_validation_report.json").read_text())
    assert report["numeric_failure_count"] == 0
    assert report["validation_ok"] is True

    # Confirma que el salvage V2 REALMENTE se ejecutó (no que el LLM
    # simplemente no incluyó el número en un segundo intento normal).
    attempt_dir = e.out / "raw_section_outputs" / "agent_attempt_01"
    salvage_files = list(attempt_dir.glob("*numeric_salvage*"))
    assert salvage_files, "se esperaba al menos un artefacto de salvage V2"


@scenario("NV2-04. Salvage imposible (una sola oración con número no respaldado, eliminarla vaciaría la sección) -> retry LLM, previous error codes llegan al prompt")
def test_nv2_04_salvage_impossible_triggers_retry_with_error_codes_in_prompt():
    responses = [
        json.dumps({"section_id": "S1", "sentences": [{"text": "El modelo alcanza un 77% de precision, un valor no reportado.", "supporting_evidence_ids": ["E1"]}]}),
        json.dumps({"section_id": "S1", "sentences": [{"text": "El modelo alcanza una precision estable en el conjunto de prueba evaluado.", "supporting_evidence_ids": ["E1"]}]}),
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
    result = e.agent.execute(_v2_ai(e))
    assert calls["n"] == 2  # fail-closed: no salvage posible (una sola oración) -> retry real
    assert len(seen_prompts) == 2
    assert "UNSUPPORTED_NUMERIC_VALUE:77%" in seen_prompts[1]
    assert result.quality_status.value == "APPROVED"


@scenario("NV2-05. Retries agotados por número no soportado -> NEEDS_REVISION/RETRY (intento 1) o HALT_STAGE (intento 2), NUNCA sección falsamente válida")
def test_nv2_05_exhausted_retries_never_falsely_valid():
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [{"text": "El modelo alcanza un 77% de precision, un valor no reportado.", "supporting_evidence_ids": ["E1"]}]})

    e1 = Env(attempt=1)
    e1.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e1.collection))
    result1 = e1.agent.execute(_v2_ai(e1))
    assert result1.execution_status.value == "COMPLETED"
    assert result1.quality_status.value == "NEEDS_REVISION"
    assert result1.requested_transition.action.value == "RETRY"
    assert not (e1.out / "state_of_art_draft.json").exists()
    report1 = json.loads((e1.out / "draft_validation_report.json").read_text())
    assert any("UNSUPPORTED_NUMERIC_VALUE:77%" in str(err) for err in report1["last_attempt_errors"])

    e2 = Env(attempt=2)
    e2.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e2.collection))
    result2 = e2.agent.execute(_v2_ai(e2))
    assert result2.quality_status.value == "NEEDS_REVISION"
    assert result2.requested_transition.action.value == "HALT_STAGE"
    assert not (e2.out / "state_of_art_draft.json").exists()


@scenario("NV2-06. Múltiples números no soportados en una oración -> salvage determinista los elimina todos juntos, comportamiento correcto")
def test_nv2_06_multiple_numbers_in_one_sentence_deterministic():
    def invoke(p):
        return json.dumps({
            "section_id": "S1",
            "sentences": [
                {"text": "El estudio de 1975 reporta un 19% de mejora con una muestra de 54 participantes.", "supporting_evidence_ids": ["E1"]},
                {"text": "El segundo hallazgo confirma la consistencia general de los resultados obtenidos.", "supporting_evidence_ids": ["E1"]},
            ],
        })

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert result.quality_status.value == "APPROVED"
    draft = json.loads((e.out / "state_of_art_draft.json").read_text())
    s1 = next(s for s in draft["sections"] if s["section_id"] == "S1")
    for value in ("1975", "19%", "54"):
        assert value not in s1["draft_text"]
    report = json.loads((e.out / "draft_validation_report.json").read_text())
    assert report["numeric_failure_count"] == 0

    attempt_dir = e.out / "raw_section_outputs" / "agent_attempt_01"
    salvage_files = list(attempt_dir.glob("*numeric_salvage*validation*"))
    assert salvage_files
    salvage_payload = json.loads(salvage_files[0].read_text())
    assert set(salvage_payload["removed_unsupported_numeric_values"]) == {"1975", "19%", "54"}


@scenario("NV2-07. Números científicamente respaldados (0.38, 19%, 1975, 10,000) no se eliminan si aparecen en la evidencia citada")
def test_nv2_07_supported_scientific_numbers_never_removed():
    evidence = [{
        "source_filename": "a.pdf", "chunk_id": "c1",
        "text": "El estudio de 1975 con 10,000 participantes reporto un 19% de mejora y una desviacion de 0.38.",
    }]
    sentences = [{
        "sentence_id": 0,
        "text": "El estudio de 1975 con 10,000 participantes reporto un 19% de mejora y una desviacion de 0.38.",
        "supporting_citations": ["[a.pdf | c1]"],
    }]
    errors = v2_numeric_support_errors(sentences, evidence)
    assert errors == []
    # También confirma que el salvage nunca se activa (nada que eliminar) para estos valores.
    assert v2_numeric_salvage(sentences, evidence) is None


@scenario("NV2-08. Evidence handle válido pero número ausente del chunk citado -> numeric error real")
def test_nv2_08_valid_handle_but_number_absent_from_chunk():
    evidence = [{"source_filename": "a.pdf", "chunk_id": "c1", "text": "El modelo alcanzo un desempeño consistente en las pruebas."}]
    sentences = [{"sentence_id": 0, "text": "El modelo alcanza un 91% de precision en el conjunto evaluado.", "supporting_citations": ["[a.pdf | c1]"]}]
    errors = v2_numeric_support_errors(sentences, evidence)
    assert errors == ["UNSUPPORTED_NUMERIC_VALUE:91%"]


@scenario("NV2-09. Legacy conserva EXACTAMENTE su comportamiento previo (misma semántica: soportado por cita propia, no soportado, perdonado por evidencia de sección)")
def test_nv2_09_legacy_behavior_unchanged():
    section = {"section_id": "S1"}

    generated_supported = {
        "section_id": "S1", "section_title": "T",
        "draft_text": "El modelo alcanza un 91% de precision en el conjunto de prueba evaluado completo [a.pdf | c1].",
        "claims": [{"claim": "El modelo alcanza un 91% de precision en el conjunto de prueba evaluado completo", "supporting_citations": ["[a.pdf | c1]"]}],
    }
    evidence = [{"source_filename": "a.pdf", "chunk_id": "c1", "text": "El modelo alcanzo 91% en las pruebas realizadas durante el estudio completo."}]
    r1 = validate_generated_section(generated_supported, section, evidence)
    assert r1["validation_ok"] is True
    assert r1["numeric_errors"] == []

    generated_unsupported = dict(generated_supported)
    generated_unsupported["draft_text"] = "El modelo alcanza un 77% de precision en el conjunto de prueba evaluado completo [a.pdf | c1]."
    generated_unsupported["claims"] = [{"claim": "El modelo alcanza un 77% de precision en el conjunto de prueba evaluado completo", "supporting_citations": ["[a.pdf | c1]"]}]
    r2 = validate_generated_section(generated_unsupported, section, evidence)
    assert r2["validation_ok"] is False
    assert r2["numeric_errors"] == ["UNSUPPORTED_NUMERIC_VALUE:77%"]

    generated_forgiven = dict(generated_supported)
    generated_forgiven["draft_text"] = "El modelo alcanza un 88% de precision en el conjunto de prueba evaluado completo [a.pdf | c1]."
    generated_forgiven["claims"] = [{"claim": "El modelo alcanza un 88% de precision en el conjunto de prueba evaluado completo", "supporting_citations": ["[a.pdf | c1]"]}]
    evidence_extra = evidence + [{"source_filename": "a.pdf", "chunk_id": "c2", "text": "En otro experimento independiente se obtuvo un 88% de acierto general."}]
    r3 = validate_generated_section(generated_forgiven, section, evidence_extra)
    assert r3["validation_ok"] is True
    assert r3["numeric_errors"] == []


@scenario("NV2-10. Integración real V2 -> build_draft_reports: todas las secciones aceptadas (con o sin salvage) producen numeric_failure_count=0")
def test_nv2_10_full_v2_integration_zero_global_numeric_failures():
    e = Env(attempt=1)
    outline_path = e.inp / "outline.json"
    outline = json.loads(outline_path.read_text())
    outline["sections"] = [
        {"section_id": "S1", "section_title": "Methods", "section_type": "linea_tematica", "purpose": "p", "key_arguments": ["a"], "evidence_needs": ["e"], "papers_to_use": [{"source_filename": "a.pdf", "title": "A"}]},
        {"section_id": "S2", "section_title": "Results", "section_type": "linea_tematica", "purpose": "p", "key_arguments": ["a"], "evidence_needs": ["e"], "papers_to_use": [{"source_filename": "a.pdf", "title": "A"}]},
    ]
    outline_path.write_text(json.dumps(outline), encoding="utf-8")
    (e.inp / "mapping.csv").write_text("section_id,source_filename,title\nS1,a.pdf,A\nS2,a.pdf,A\n")

    def invoke(p):
        if '"S1"' in p:
            # S1: numero no soportado -> requiere salvage
            return json.dumps({
                "section_id": "S1",
                "sentences": [
                    {"text": "El modelo alcanza un 77% de precision, un valor no reportado por la evidencia.", "supporting_evidence_ids": ["E1"]},
                    {"text": "El segundo hallazgo confirma la consistencia general de los resultados obtenidos.", "supporting_evidence_ids": ["E1"]},
                ],
            })
        # S2: camino normal, sin numeros problematicos
        return json.dumps({
            "section_id": "S2",
            "sentences": [{"text": "El tercer hallazgo replica el patron observado en el estudio anterior de forma consistente.", "supporting_evidence_ids": ["E1"]}],
        })

    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e, extra_policy={"min_total_words": 1}))
    report = json.loads((e.out / "draft_validation_report.json").read_text())
    assert report["numeric_failure_count"] == 0
    assert report["all_section_validations_ok"] is True
    assert result.quality_status.value == "APPROVED"
    assert report["validation_ok"] is True

    # Confirma también contra build_draft_reports directamente, sobre
    # las secciones REALMENTE aceptadas por el flujo V2 real.
    draft = json.loads((e.out / "state_of_art_draft.json").read_text())
    assert "77%" not in json.dumps(draft)


@scenario("NV2-11. Handle citado (E1) sin soporte + evidencia recuperada NO citada (E2) SÍ contiene el valor -- V2 nunca perdona por E2, produce numeric error, salvage/retry lo resuelve, sección final aceptada no contiene el valor sin soporte citado, numeric_failure_count global = 0")
def test_nv2_11_v2_never_forgives_via_uncited_retrieved_evidence():
    # Verificación PURA del núcleo del comportamiento primero: E1 (chunk
    # citado) NO contiene "54"; E2 (chunk recuperado para la sección,
    # pero NO citado por esta oración) SÍ lo contiene.
    evidence = [
        {"source_filename": "paper_e1.pdf", "chunk_id": "c1", "text": "El estudio describe la metodologia general aplicada sin detallar conteos especificos."},
        {"source_filename": "paper_e2.pdf", "chunk_id": "c2", "text": "Se registraron 54 casos en el grupo de control durante el seguimiento."},
    ]
    sentences = [{
        "sentence_id": 0,
        "text": "Se observaron 54 casos durante el periodo de seguimiento evaluado en el estudio.",
        "supporting_citations": ["[paper_e1.pdf | c1]"],  # cita SOLO E1 -- nunca E2
    }]
    errors = v2_numeric_support_errors(sentences, evidence)
    assert errors == ["UNSUPPORTED_NUMERIC_VALUE:54"], "V2 NO debe perdonar el 54 solo porque E2 (no citado) lo contiene"

    # Flujo real end-to-end: usando los chunks reales del fixture
    # estándar (E1=paper_a.pdf|a_chroma con "91%", E2=paper_a.pdf|
    # a_shared con "92%", ambos recuperados para la sección) -- el LLM
    # cita SOLO E1 pero usa el valor "92" (que únicamente aparece en
    # E2, no citado por esta oración).
    def invoke(p):
        return json.dumps({
            "section_id": "S1",
            "sentences": [
                {"text": "El metodo alcanza un 92% de precision segun el analisis realizado en el estudio.", "supporting_evidence_ids": ["E1"]},
                {"text": "El segundo hallazgo confirma la consistencia general de los resultados obtenidos en el estudio.", "supporting_evidence_ids": ["E1"]},
            ],
        })

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e))
    assert result.quality_status.value == "APPROVED"

    draft = json.loads((e.out / "state_of_art_draft.json").read_text())
    s1 = next(s for s in draft["sections"] if s["section_id"] == "S1")
    # La sección final aceptada NO contiene el valor sin soporte citado
    # -- el salvage lo eliminó (oración con "92%" removida).
    assert "92%" not in s1["draft_text"]
    assert all("92%" not in c["claim"] for c in s1["claims"])

    report = json.loads((e.out / "draft_validation_report.json").read_text())
    assert report["numeric_failure_count"] == 0

    attempt_dir = e.out / "raw_section_outputs" / "agent_attempt_01"
    assert list(attempt_dir.glob("*numeric_salvage*")), "se esperaba evidencia de salvage V2"


@scenario("NV2-11b. Comprobación explícita: legacy SÍ perdona en el MISMO escenario (soporte en otra evidencia de sección) -- comportamiento histórico preservado")
def test_nv2_11b_legacy_still_forgives_via_section_evidence():
    evidence = [
        {"source_filename": "paper_e1.pdf", "chunk_id": "c1", "text": "El estudio describe la metodologia general aplicada sin detallar conteos especificos."},
        {"source_filename": "paper_e2.pdf", "chunk_id": "c2", "text": "Se registraron 54 casos en el grupo de control durante el seguimiento."},
    ]
    generated = {
        "section_id": "S1", "section_title": "T",
        "draft_text": "Se observaron 54 casos durante el periodo de seguimiento evaluado en el estudio [paper_e1.pdf | c1].",
        "claims": [{"claim": "Se observaron 54 casos durante el periodo de seguimiento evaluado en el estudio", "supporting_citations": ["[paper_e1.pdf | c1]"]}],
    }
    section = {"section_id": "S1"}
    r = validate_generated_section(generated, section, evidence)
    assert r["validation_ok"] is True
    assert r["numeric_errors"] == []


if __name__ == "__main__":
    for fn in (
        test_nv2_01_supported_number_no_salvage_needed,
        test_nv2_02_unsupported_number_detected_before_acceptance,
        test_nv2_03_salvage_success_produces_clean_accepted_section,
        test_nv2_04_salvage_impossible_triggers_retry_with_error_codes_in_prompt,
        test_nv2_05_exhausted_retries_never_falsely_valid,
        test_nv2_06_multiple_numbers_in_one_sentence_deterministic,
        test_nv2_07_supported_scientific_numbers_never_removed,
        test_nv2_08_valid_handle_but_number_absent_from_chunk,
        test_nv2_09_legacy_behavior_unchanged,
        test_nv2_10_full_v2_integration_zero_global_numeric_failures,
        test_nv2_11_v2_never_forgives_via_uncited_retrieved_evidence,
        test_nv2_11b_legacy_still_forgives_via_section_evidence,
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
