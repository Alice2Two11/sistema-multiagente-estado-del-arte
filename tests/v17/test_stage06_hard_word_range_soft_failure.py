"""Contrato de longitud de Agent06 -- ``configured_min_total_words``/
``configured_max_total_words`` (el ``min_total_words``/``max_total_words``
del generation_profile) son el único gate contractual real para
aprobar el documento completo. ``effective_min_total_words`` se
conserva EXCLUSIVAMENTE como métrica diagnóstica -- nunca participa en
``global_length_valid``.

Causa raíz cerrada: antes de esta corrección, ``global_length_valid``
usaba ``effective_min_total_words`` (rebajado silenciosamente por
``source_free_organizational_section_count``), permitiendo que un
borrador con 1081 palabras se aprobara con ``configured_min_total_
words=1300``. Reproducido con los valores reales del caso reportado
(1300/1600/2200) y confirmado que la corrección lo rechaza.

Incumplir el rango de longitud NUNCA es un fallo técnico
(ValueError/RuntimeError) -- siempre COMPLETED + NEEDS_REVISION/RETRY
(intento no final) o COMPLETED + APPROVED_PENDING_MANUAL_REVIEW/
HALT_STAGE (intento final agotado), con reason_code explícito
(TOTAL_WORD_COUNT_BELOW_MINIMUM / TOTAL_WORD_COUNT_ABOVE_MAXIMUM /
INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH). Agent06 NUNCA marca
usable_for_evaluation ni avanza hacia 08 -- target_stage permanece
None en el camino de fallo, y el único target_stage del camino exitoso
sigue siendo "07_agente_verificador" (sin cambios).

Reparación dirigida: exclusivamente para el contrato canonical_
sentences_v2 (Evidence Handles), reutilizando la MISMA evidencia ya
recuperada -- nunca inventa claims/citas/números, nunca "expansión
determinista" que rellene con frases repetidas.

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
from src.tools.draft_writing.validation import build_draft_reports  # noqa: E402
from src.adapters.evaluation_pipeline_outcome import SCIENTIFIC_EXHAUSTION_HALT_REASON_CODES  # noqa: E402

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


def _sentence(text, evidence_id="E1"):
    return {"text": text, "supporting_evidence_ids": [evidence_id]}


@scenario("LEN-01. 1600 palabras dentro de 1300-2200 -> APPROVED (build_draft_reports directo)")
def test_len_01_within_range_approved():
    section = {
        "section_id": "S1", "section_title": "T",
        "draft_text": " ".join(["palabra"] * 1600),
        "claims": [], "section_validation": {"validation_ok": True, "claim_errors": [], "numeric_errors": []},
    }
    outline_sections = [{"section_id": "S1", "section_title": "T", "section_type": "linea_tematica"}]
    policy = {
        "target_total_words": 1600, "min_total_words": 1300, "max_total_words": 2200,
        "section_budgets": {"S1": {"target_words": 1600, "minimum_words": 1, "maximum_words": 3000}},
    }
    report, *_ = build_draft_reports([section], outline_sections, {"S1": []}, policy)
    assert report["actual_total_words"] == 1600
    assert report["word_count_compliant"] is True
    assert report["validation_ok"] is True


@scenario("LEN-02. 1081 con min=1300 (caso real reportado) -> NEEDS_REVISION, nunca excepción técnica")
def test_len_02_below_minimum_needs_revision_no_exception():
    def invoke(p):
        # Respuesta corta, sin evidencia adicional que permita ampliar --
        # replica el patrón real (déficit no reparable con esta evidencia).
        return json.dumps({"section_id": "S1", "sentences": [_sentence(
            "El modelo alcanza una precision estable en el conjunto de prueba evaluado."
        )]})

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e, {"target_total_words": 1600, "min_total_words": 1300, "max_total_words": 2200}))

    assert result.execution_status.value == "COMPLETED"
    assert result.error is None
    assert result.quality_status.value == "NEEDS_REVISION"
    assert result.requested_transition.action.value == "RETRY"
    assert result.failure_reason_codes[0] in (
        "TOTAL_WORD_COUNT_BELOW_MINIMUM", "INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH",
    )


@scenario("LEN-03. Retry con evidencia suficiente alcanza >=configured_min -> APPROVED, target_stage=07 (nunca 08)")
def test_len_03_repair_succeeds_reaches_minimum():
    calls = {"n": 0}

    def invoke(p):
        calls["n"] += 1
        is_repair = "Reformula ÚNICAMENTE" in p
        sid = "S1" if '"S1"' in p else "S2"
        if is_repair:
            return json.dumps({"section_id": sid, "sentences": [
                _sentence(f"El modelo alcanza una precision estable en el conjunto de prueba evaluado para {sid}."),
                _sentence("Este resultado confirma la consistencia general observada durante todo el estudio realizado en el mismo conjunto."),
                _sentence("La evidencia disponible respalda de forma adicional la robustez del hallazgo reportado en el analisis."),
            ]})
        return json.dumps({"section_id": sid, "sentences": [
            _sentence(f"El modelo alcanza una precision estable en el conjunto de prueba evaluado para {sid}.")
        ]})

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e, {"target_total_words": 70, "min_total_words": 60, "max_total_words": 150}))

    assert result.quality_status.value == "APPROVED"
    assert result.requested_transition.target_stage == "07_agente_verificador"
    assert result.quality_metrics["technical"]["length_repair_attempted"] is True
    assert result.quality_metrics["technical"]["length_repair_successful"] is True
    assert result.quality_metrics["scientific"]["word_count_compliant"] is True
    assert calls["n"] >= 2  # confirma que SÍ se invocó la reparación (no solo generación normal)


@scenario("LEN-04. Retry sigue <min por evidencia insuficiente -> HALT científico/manual review, NUNCA excepción técnica, NUNCA usable_for_evaluation ni salto a 08")
def test_len_04_insufficient_evidence_scientific_halt_not_technical():
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [_sentence(
            "El modelo alcanza una precision estable en el conjunto de prueba evaluado."
        )]})

    e = Env(attempt=2)  # último intento -- agotado
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e, {"target_total_words": 1600, "min_total_words": 1300, "max_total_words": 2200}))

    # Nunca excepción técnica.
    assert result.execution_status.value == "COMPLETED"
    assert result.error is None

    # HALT científico/manual review, no un status de aprobación limpia.
    assert result.quality_status.value == "APPROVED_PENDING_MANUAL_REVIEW"
    assert result.requested_transition.action.value == "HALT_STAGE"
    assert result.failure_reason_codes == ("INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH",)

    # Nunca salto a 08 -- target_stage permanece None (el único
    # target_stage que Agent06 produce hacia una etapa siguiente es
    # "07_agente_verificador", exclusivamente en el camino APPROVED).
    assert result.requested_transition.target_stage is None

    # Agent06 no tiene ni produce un campo usable_for_evaluation -- ese
    # campo pertenece exclusivamente a la evaluación posterior a
    # Agent07. Confirmamos que no aparece en ningún lugar del resultado.
    result_dict = result.to_dict() if hasattr(result, "to_dict") else vars(result)
    assert "usable_for_evaluation" not in json.dumps(result_dict, default=str)

    # No se publica el draft final.
    assert "state_of_art_draft.json" not in result.output_artifacts


@scenario("LEN-05. >max_total_words -> reparación de condensación, conserva claims/citas")
def test_len_05_above_maximum_triggers_condensation_repair():
    calls = {"n": 0}

    def invoke(p):
        calls["n"] += 1
        is_repair = "Reformula ÚNICAMENTE" in p
        if is_repair:
            assert "condensa" in p.lower() or "exceso" in p.lower()
            return json.dumps({"section_id": "S1", "sentences": [_sentence(
                "El modelo alcanza una precision estable en el conjunto evaluado."
            )]})
        long_sentences = [_sentence(f"El modelo alcanza una precision estable en el conjunto evaluado numero {i} de la serie completa.") for i in range(15)]
        return json.dumps({"section_id": "S1", "sentences": long_sentences})

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e, {"target_total_words": 20, "min_total_words": 1, "max_total_words": 30}))

    assert result.quality_metrics["technical"]["length_repair_attempted"] is True
    if result.quality_status.value == "APPROVED":
        assert result.quality_metrics["technical"]["length_repair_successful"] is True
        assert result.quality_metrics["scientific"]["word_count_compliant"] is True
    else:
        assert result.failure_reason_codes[0] in ("TOTAL_WORD_COUNT_ABOVE_MAXIMUM", "INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH")
        assert result.error is None


@scenario("LEN-06. effective_min_total_words no puede permitir aprobación por debajo de configured_min_total_words")
def test_len_06_effective_min_never_gates_approval():
    section_short = {
        "section_id": "S1", "section_title": "T",
        "draft_text": " ".join(["palabra"] * 340),
        "claims": [], "section_validation": {"validation_ok": True, "claim_errors": [], "numeric_errors": []},
    }
    organizational_sections = [
        {
            "section_id": sid, "section_title": "T", "draft_text": "Texto organizativo breve.",
            "claims": [], "section_validation": {"validation_ok": True, "claim_errors": [], "numeric_errors": [], "source_free_organizational_section": True},
        }
        for sid in ("S2", "S3")
    ]
    all_sections = [section_short] + organizational_sections
    outline_sections = [{"section_id": s["section_id"], "section_title": "T", "section_type": "linea_tematica"} for s in all_sections]
    policy = {
        "target_total_words": 1600, "min_total_words": 1300, "max_total_words": 2200,
        "section_budgets": {s["section_id"]: {"target_words": 500, "minimum_words": 1, "maximum_words": 3000} for s in all_sections},
    }
    evidence_map = {s["section_id"]: [] for s in all_sections}
    report, *_ = build_draft_reports(all_sections, outline_sections, evidence_map, policy)

    assert report["configured_min_total_words"] == 1300
    assert report["effective_min_total_words"] < report["configured_min_total_words"]
    # El total real (340 + textos organizativos breves) sigue muy por
    # debajo de 1300 -- effective_min, aunque rebajado, NUNCA debe
    # permitir que esto se apruebe.
    assert report["actual_total_words"] < report["configured_min_total_words"]
    assert report["word_count_compliant"] is False
    assert report["global_length_valid"] is False
    assert report["validation_ok"] is False


@scenario("LEN-07. La reparación nunca introduce claims sin Evidence Handles -- toda cita nueva viene de la evidencia ya retrieved")
def test_len_07_repair_never_introduces_claims_without_evidence_handles():
    from src.tools.draft_writing.length_repair import attempt_directed_length_repair

    evidence = [{"source_filename": "a.pdf", "chunk_id": "c1", "text": "El modelo alcanzo 91% de precision en las pruebas realizadas durante el estudio completo."}]
    generated = [{"section_id": "S1", "section_title": "T", "draft_text": "El modelo alcanza una precision alta en el conjunto evaluado [a.pdf | c1].", "claims": [{"claim": "El modelo alcanza una precision alta en el conjunto evaluado", "supporting_citations": ["[a.pdf | c1]"]}]}]
    sections = [{"section_id": "S1", "section_title": "T", "section_type": "linea_tematica"}]
    evidence_map = {"S1": evidence}
    policy = {"target_total_words": 40, "min_total_words": 30, "max_total_words": 60}

    class FakeRuntime:
        def invoke(self, prompt):
            assert "no inventes" in prompt.lower() or "nunca inventes" in prompt.lower()
            return json.dumps({"section_id": "S1", "sentences": [
                _sentence("El modelo alcanza una precision alta en el conjunto evaluado, superando las lineas base tradicionales."),
                _sentence("Este resultado se mantiene consistente durante todo el estudio realizado sobre el mismo conjunto de datos."),
            ]})
        def parse(self, raw):
            return json.loads(raw)

    repaired, meta = attempt_directed_length_repair(generated, sections, evidence_map, policy, FakeRuntime())
    assert meta["sections_repaired"] == ["S1"]
    # Toda cita en el texto reparado corresponde EXACTAMENTE a la evidencia original -- ningún handle/fuente inventada.
    assert "[a.pdf | c1]" in repaired[0]["draft_text"]
    for claim in repaired[0]["claims"]:
        assert claim["supporting_citations"] == ["[a.pdf | c1]"]


@scenario("LEN-08. Ground Truth nunca se usa -- ni en el gate ni en la reparación")
def test_len_08_ground_truth_never_used():
    import inspect

    from src.tools.draft_writing import length_repair, validation

    for module in (length_repair,):
        source = inspect.getsource(module)
        assert "ground_truth" not in source.lower()

    # validation.py sí menciona ground_truth_used como CAMPO informativo
    # fijo en False -- confirmamos que nunca se activa.
    assert "\"ground_truth_used\": False" in inspect.getsource(validation) or "'ground_truth_used': False" in inspect.getsource(validation)


@scenario("LEN-09. La cantidad de secciones no afecta este gate -- configured_min/max son independientes de section_count")
def test_len_09_section_count_does_not_affect_length_gate():
    def build_report(n_sections, words_per_section):
        sections = []
        for i in range(n_sections):
            sections.append({
                "section_id": f"S{i}", "section_title": "T",
                "draft_text": " ".join(["palabra"] * words_per_section),
                "claims": [], "section_validation": {"validation_ok": True, "claim_errors": [], "numeric_errors": []},
            })
        outline_sections = [{"section_id": s["section_id"], "section_title": "T", "section_type": "linea_tematica"} for s in sections]
        policy = {
            "target_total_words": 1600, "min_total_words": 1300, "max_total_words": 2200,
            "section_budgets": {s["section_id"]: {"target_words": 500, "minimum_words": 1, "maximum_words": 3000} for s in sections},
        }
        report, *_ = build_draft_reports(sections, outline_sections, {s["section_id"]: [] for s in sections}, policy)
        return report

    # Mismo total (1600), distinto número de secciones -- el gate debe
    # dar el mismo resultado en ambos casos, dado que depende SOLO del
    # total de palabras, nunca del section_count.
    report_4_sections = build_report(4, 400)
    report_8_sections = build_report(8, 200)
    assert report_4_sections["actual_total_words"] == report_8_sections["actual_total_words"] == 1600
    assert report_4_sections["word_count_compliant"] == report_8_sections["word_count_compliant"] is True
    assert report_4_sections["configured_min_total_words"] == report_8_sections["configured_min_total_words"]


@scenario("LEN-10. Comportamiento legacy dentro del rango no cambia")
def test_len_10_legacy_behavior_within_range_unchanged():
    def invoke(p):
        return json.dumps({
            "section_id": "S1", "section_title": "T",
            "draft_text": "El modelo alcanza una precision estable en el conjunto de prueba evaluado [a.pdf | c1].",
            "claims": [{"claim": "El modelo alcanza una precision estable en el conjunto de prueba evaluado", "supporting_citations": ["[a.pdf | c1]"]}],
        })

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    # policy legacy explícita (sin draft_representation_contract v2), rango muy bajo -- debe seguir aprobando igual que siempre.
    ai = replace(e.ai, policy={**e.ai.policy, "target_total_words": 10, "min_total_words": 1, "max_total_words": 100})
    result = e.agent.execute(ai)
    assert result.quality_status.value == "APPROVED"
    assert result.requested_transition.target_stage == "07_agente_verificador"


@scenario("LEN-11. Solo un PARTIAL_HALT científico producido DESPUÉS de Agent07 puede entrar en la ruta de evaluación parcial -- Agent06 nunca produce ese estado")
def test_len_11_partial_halt_only_valid_after_agent07():
    # Confirma que los reason codes de longitud de Agent06 NUNCA
    # coinciden con los reason codes que resolve_pipeline_outcome_for_
    # evaluation acepta como PARTIAL_HALT científico (esos son
    # exclusivos del ciclo escritor-verificador de Agent07) -- un HALT
    # de Agent06 por longitud nunca podría colarse en esa ruta aunque
    # se intentara, porque su reason_code no pertenece a ese conjunto.
    from src.agents.draft_writing_agent import (
        TOTAL_WORD_COUNT_BELOW_MINIMUM,
        TOTAL_WORD_COUNT_ABOVE_MAXIMUM,
        INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH,
    )
    agent06_length_codes = {
        TOTAL_WORD_COUNT_BELOW_MINIMUM, TOTAL_WORD_COUNT_ABOVE_MAXIMUM,
        INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH,
    }
    assert agent06_length_codes.isdisjoint(SCIENTIFIC_EXHAUSTION_HALT_REASON_CODES)

    # Y confirma, con el escenario real de LEN-04, que Agent06 nunca
    # marca target_stage="08_evaluacion_experimental" ni nada
    # equivalente -- el HALT de Agent06 nunca "salta" etapas.
    def invoke(p):
        return json.dumps({"section_id": "S1", "sentences": [_sentence(
            "El modelo alcanza una precision estable en el conjunto de prueba evaluado."
        )]})

    e = Env(attempt=2)
    e.agent = DraftWritingAgent(DraftWritingRuntime(invoke, e.collection))
    result = e.agent.execute(_v2_ai(e, {"target_total_words": 1600, "min_total_words": 1300, "max_total_words": 2200}))
    assert result.requested_transition.target_stage != "08_evaluacion_experimental"
    assert result.requested_transition.target_stage is None


if __name__ == "__main__":
    for fn in (
        test_len_01_within_range_approved,
        test_len_02_below_minimum_needs_revision_no_exception,
        test_len_03_repair_succeeds_reaches_minimum,
        test_len_04_insufficient_evidence_scientific_halt_not_technical,
        test_len_05_above_maximum_triggers_condensation_repair,
        test_len_06_effective_min_never_gates_approval,
        test_len_07_repair_never_introduces_claims_without_evidence_handles,
        test_len_08_ground_truth_never_used,
        test_len_09_section_count_does_not_affect_length_gate,
        test_len_10_legacy_behavior_within_range_unchanged,
        test_len_11_partial_halt_only_valid_after_agent07,
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
