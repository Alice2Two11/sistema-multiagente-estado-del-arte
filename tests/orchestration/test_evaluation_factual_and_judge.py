"""Pruebas del ensamblador factual (5A+5B) y del LLM Judge."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.evaluation.factual_assembly import (
    build_factual_audit,
    evaluate_factual_consistency,
    resolve_factual_gate,
)
from src.tools.evaluation.llm_judge import (
    JUDGE_CRITERIA,
    build_judge_score_rows,
    parse_json_safely,
    run_llm_judge,
    validate_judge_result,
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


def _chunks(rows):
    return [{"source_filename": s, "chunk_id": c, "text": t} for s, c, t in rows]


def _sections(rows):
    return [{"section_id": sid, "draft_text": text} for sid, text in rows]


def _valid_judge_response():
    return {
        "scores": {
            criterion: {"score": 4, "justification": "Justificación válida.", "evidence_from_generated": []}
            for criterion in JUDGE_CRITERIA
        },
        "strengths": ["Fortaleza uno."],
        "organization_differences": [],
        "missing_topics_or_omissions": [],
        "overall_assessment": "Evaluación general aceptable.",
    }


class FakeJudgeLLMFactory:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.instances_created = 0

    def __call__(self):
        self.instances_created += 1
        return self

    def invoke(self, messages):
        self.calls.append(messages[0].content)
        content = self.responses.pop(0)
        return SimpleNamespace(content=content)


# ---------------------------------------------------------------------------
# Ensamblador factual: aprobado / pendiente no bloqueante / bloqueante
# ---------------------------------------------------------------------------


@scenario("Y01. Factual consistency aprobada (todo perfecto)")
def test_factual_consistency_approved():
    sections = _sections([("s1", "El resultado fue 95 puntos [p.pdf | c1].")])
    chunks = _chunks([("p.pdf", "c1", "El resultado fue 95 puntos.")])
    traceability_rows = [
        {"claim_id": "1", "verdict": "supported", "claim": "El resultado fue 95 puntos.",
         "source_filename": "p.pdf", "chunk_id": "c1", "hallucination_risk": "low"}
    ]
    audit = build_factual_audit(
        sections=sections, chunks=chunks, traceability_rows=traceability_rows,
        generated_content_text="El resultado fue 95 puntos.",
    )
    consistency = evaluate_factual_consistency(audit)
    assert consistency["factual_consistency_ok"] is True
    assert consistency["factual_consistency_status"] == "APPROVED"
    resolve_factual_gate(
        factual_consistency_result=consistency, source_stage="AGENT07", upstream_runtime_status="COMPLETED"
    )  # no debe lanzar


@scenario("Y02. Pendientes no bloqueantes (source_stage=AGENT07, upstream PARTIAL)")
def test_pending_non_blocking():
    sections = _sections([("s1", "El resultado fue exitoso [p.pdf | c1].")])
    chunks = _chunks([("p.pdf", "c1", "Texto sin relación.")])  # no matchea -> claim_in_final_text sigue OK pero evidencia falla si aplica
    traceability_rows = [
        {"claim_id": "1", "verdict": "unsupported", "claim": "El resultado fue exitoso.",
         "hallucination_risk": "high"}
    ]
    audit = build_factual_audit(
        sections=sections, chunks=chunks, traceability_rows=traceability_rows,
        generated_content_text="El resultado fue exitoso.",
    )
    consistency = evaluate_factual_consistency(audit)
    assert consistency["factual_consistency_ok"] is False
    resolve_factual_gate(
        factual_consistency_result=consistency, source_stage="AGENT07", upstream_runtime_status="PARTIAL"
    )  # no debe lanzar (no bloqueante)


@scenario("Y03. Fallo factual bloqueante (upstream no es AGENT07/PARTIAL)")
def test_blocking_failure():
    sections = _sections([("s1", "El resultado fue exitoso [p.pdf | c1].")])
    chunks = _chunks([("p.pdf", "c1", "Texto sin relación.")])
    traceability_rows = [
        {"claim_id": "1", "verdict": "unsupported", "claim": "El resultado fue exitoso.",
         "hallucination_risk": "high"}
    ]
    audit = build_factual_audit(
        sections=sections, chunks=chunks, traceability_rows=traceability_rows,
        generated_content_text="El resultado fue exitoso.",
    )
    consistency = evaluate_factual_consistency(audit)
    try:
        resolve_factual_gate(
            factual_consistency_result=consistency, source_stage="AGENT07", upstream_runtime_status="COMPLETED"
        )
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:
        raise AssertionError("debía lanzar ValueError bloqueante")


@scenario("Y04. factual_consistency_ok=False cuando no hay citas ni números (None no cuenta como 0.0)")
def test_no_citations_or_numbers_never_approved():
    sections = _sections([("s1", "Un texto perfecto sin citas ni números.")])
    chunks = _chunks([])
    traceability_rows = [{"claim_id": "1", "verdict": "supported", "claim": "Un texto perfecto sin citas ni números."}]
    audit = build_factual_audit(
        sections=sections, chunks=chunks, traceability_rows=traceability_rows,
        generated_content_text="Un texto perfecto sin citas ni números.",
    )
    consistency = evaluate_factual_consistency(audit)
    assert audit["citation_metrics"]["citation_error_rate"] is None
    assert audit["numeric_metrics"]["numeric_error_rate"] is None
    assert consistency["factual_consistency_ok"] is False  # None nunca cuenta como 0.0


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------


@scenario("Z01. LLM Judge válido en el primer intento")
def test_llm_judge_valid_first_attempt():
    import json as _json

    factory = FakeJudgeLLMFactory([_json.dumps(_valid_judge_response())])
    result = run_llm_judge(
        topic_name="Tema de prueba", source_stage="AGENT07",
        automatic_metrics={"rougeL_fmeasure": 0.8}, factual_metrics={"factual_precision": 1.0},
        generated_plain_text="Texto generado.", ground_truth_plain_text="Texto de referencia.",
        max_generated_chars=1000, max_ground_truth_chars=1000, max_attempts=3,
        llm_factory=factory,
    )
    assert result["judge_mode"] == "new"
    assert factory.instances_created == 1
    rows = build_judge_score_rows(result["result"])
    assert len(rows) == len(JUDGE_CRITERIA)


@scenario("Z02. JSON inválido en el primer intento, válido en el segundo (reintento real)")
def test_llm_judge_invalid_json_then_valid():
    import json as _json

    factory = FakeJudgeLLMFactory(["esto no es json", _json.dumps(_valid_judge_response())])
    result = run_llm_judge(
        topic_name="Tema", source_stage="AGENT07", automatic_metrics={}, factual_metrics={},
        generated_plain_text="G.", ground_truth_plain_text="GT.",
        max_generated_chars=100, max_ground_truth_chars=100, max_attempts=3,
        llm_factory=factory,
    )
    assert factory.instances_created == 2
    assert result["result"]["overall_assessment"]


@scenario("Z03. Agotamiento de intentos: siempre inválido -> ValueError con los errores acumulados")
def test_llm_judge_exhaustion():
    factory = FakeJudgeLLMFactory(["invalido"] * 3)
    try:
        run_llm_judge(
            topic_name="Tema", source_stage="AGENT07", automatic_metrics={}, factual_metrics={},
            generated_plain_text="G.", ground_truth_plain_text="GT.",
            max_generated_chars=100, max_ground_truth_chars=100, max_attempts=3,
            llm_factory=factory,
        )
    except ValueError as exc:
        assert "3 intentos" in str(exc)
    else:
        raise AssertionError("debía agotar los 3 intentos y lanzar")
    assert factory.instances_created == 3


@scenario("Z04. Fallo del LLM (excepción en .invoke) se propaga sin silenciarse")
def test_llm_judge_llm_failure_propagates():
    class FailingFactory:
        instances_created = 0

        def __call__(self):
            self.instances_created += 1
            return self

        def invoke(self, messages):
            raise RuntimeError("Fallo simulado del LLM Judge.")

    factory = FailingFactory()
    try:
        run_llm_judge(
            topic_name="Tema", source_stage="AGENT07", automatic_metrics={}, factual_metrics={},
            generated_plain_text="G.", ground_truth_plain_text="GT.",
            max_generated_chars=100, max_ground_truth_chars=100, max_attempts=3,
            llm_factory=factory,
        )
    except RuntimeError as exc:
        assert "simulado" in str(exc)
    else:
        raise AssertionError("debía propagar el fallo del LLM")


@scenario("Z05. validate_judge_result detecta score fuera de rango, evidencia > 20 palabras, criterio faltante")
def test_validate_judge_result_errors():
    bad = _valid_judge_response()
    bad["scores"]["coherence"]["score"] = 7  # fuera de 1-5
    bad["scores"]["organization"]["evidence_from_generated"] = ["palabra " * 25]  # > 20 palabras
    del bad["scores"]["critical_depth"]  # criterio faltante
    errors = validate_judge_result(bad)
    assert any("score_not_integer_1_to_5" in e for e in errors)
    assert any("evidence_item_over_20_words" in e for e in errors)
    assert any("missing_criteria" in e for e in errors)


@scenario("Z06. parse_json_safely extrae JSON envuelto en bloque markdown")
def test_parse_json_safely_markdown_block():
    text = '```json\n{"a": 1}\n```'
    assert parse_json_safely(text) == {"a": 1}


@scenario("Z07. build_judge_score_rows: 5 filas, una por criterio, con score entero")
def test_judge_score_rows_shape():
    rows = build_judge_score_rows(_valid_judge_response())
    assert len(rows) == 5
    assert all(isinstance(r["score_1_to_5"], int) for r in rows)


if __name__ == "__main__":
    for fn in (
        test_factual_consistency_approved,
        test_pending_non_blocking,
        test_blocking_failure,
        test_no_citations_or_numbers_never_approved,
        test_llm_judge_valid_first_attempt,
        test_llm_judge_invalid_json_then_valid,
        test_llm_judge_exhaustion,
        test_llm_judge_llm_failure_propagates,
        test_validate_judge_result_errors,
        test_parse_json_safely_markdown_block,
        test_judge_score_rows_shape,
    ):
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
