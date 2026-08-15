"""J08 -- feedback de retry ESTRUCTURADO y específico para el LLM
Judge de Agent08, reemplazando el truncado mecánico que se propuso
inicialmente y fue rechazado (podía eliminar negaciones, contrastes,
condiciones o calificadores del significado). Cuando un
``evidence_item`` excede 20 palabras, el siguiente intento recibe la
ubicación exacta (``criterion``/``evidence_item_index``), el conteo
real y el límite -- pidiéndole al modelo que REFORMULE, nunca que se
le recorte el texto por él. Nunca hay repair/aceptación silenciosa: el
único camino para una respuesta con errores sigue siendo el retry LLM
real, hasta agotar los intentos (fail-closed).

``EVIDENCE_ITEM_MAX_WORDS`` se mantiene en 20 (mismo límite del
prompt). El ``error_code`` público de ``validate_judge_result``
permanece exactamente igual (``"{criterion}:evidence_item_over_20_
words"``) -- el feedback estructurado vive aparte
(``evidence_length_violations``/``build_retry_feedback``), sin romper
ningún consumidor existente de ese código.

Este cambio vive exclusivamente en ``llm_judge.py`` -- no toca
Agent06/07, el orquestador, ``evaluation_pipeline_outcome.py``
(PARTIAL_HALT), Ground Truth, ni ninguna métrica automática/factual.

Multidominio y genérico: ningún test usa contenido de un experimento
real."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import AIMessage  # noqa: E402

from src.tools.evaluation.llm_judge import (  # noqa: E402
    EVIDENCE_ITEM_MAX_WORDS,
    JUDGE_CRITERIA,
    build_retry_feedback,
    evidence_length_violations,
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


class FakeJudgeLLM:
    """VerificationLLM real (protocolo invoke) -- respuestas
    programadas, sin mocks del resto del flujo de run_llm_judge."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        content = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return AIMessage(content=content)


def _score_item(score=4, justification="Justificación académica concreta.", evidence=None):
    return {"score": score, "justification": justification, "evidence_from_generated": evidence or []}


def _full_result(argumentative_evidence, overrides=None):
    scores = {criterion: _score_item() for criterion in JUDGE_CRITERIA}
    scores["argumentative_clarity"] = _score_item(evidence=argumentative_evidence)
    if overrides:
        for criterion, patch in overrides.items():
            scores[criterion].update(patch)
    return {
        "scores": scores,
        "strengths": [],
        "organization_differences": [],
        "missing_topics_or_omissions": [],
        "overall_assessment": "Evaluación general sólida y consistente con la rúbrica.",
    }


SHORT_EVIDENCE = ["El modelo mejora respecto a la línea base evaluada en el estudio realizado."]  # <=20 palabras
LONG_EVIDENCE = [
    "El modelo comparado muestra una mejora consistente respecto a las líneas base evaluadas "
    "en el estudio realizado durante todo el período experimental completo."
]  # >20 palabras


def _run_judge(responses, max_attempts=3):
    llm = FakeJudgeLLM(responses)
    out = run_llm_judge(
        topic_name="Tema sintético multidominio",
        source_stage="06_agente_redactor",
        automatic_metrics={},
        factual_metrics={},
        generated_plain_text="Texto generado sintético de prueba, sin relación con ningún experimento real. " * 10,
        ground_truth_plain_text="Texto de referencia sintético de prueba, tampoco de ningún experimento real. " * 10,
        max_generated_chars=4000,
        max_ground_truth_chars=4000,
        max_attempts=max_attempts,
        llm_factory=lambda: llm,
    )
    return llm, out


@scenario("J08-01. evidence_item <=20 palabras -> válido, una sola llamada LLM")
def test_j08_01_short_evidence_valid_single_call():
    assert len(SHORT_EVIDENCE[0].split()) <= EVIDENCE_ITEM_MAX_WORDS
    llm, out = _run_judge([json.dumps(_full_result(SHORT_EVIDENCE))])
    assert llm.calls == 1
    assert validate_judge_result(out["result"]) == []
    assert out["attempt_errors"] == [[]]


@scenario("J08-02. evidence_item >20 palabras -> el error identifica criterion/evidence_item_index/actual_word_count/max_word_count=20")
def test_j08_02_error_identifies_criterion_index_count():
    result = _full_result(LONG_EVIDENCE)
    actual_count = len(LONG_EVIDENCE[0].split())
    assert actual_count > EVIDENCE_ITEM_MAX_WORDS

    violations = evidence_length_violations(result)
    assert len(violations) == 1
    violation = violations[0]
    assert violation["criterion"] == "argumentative_clarity"
    assert violation["evidence_item_index"] == 0
    assert violation["actual_word_count"] == actual_count
    assert violation["max_word_count"] == EVIDENCE_ITEM_MAX_WORDS
    assert str(actual_count) in violation["message"]
    assert "evidence_from_generated[0]" in violation["message"]
    assert "reformula" in violation["message"].lower()

    # El error_code público (validate_judge_result) permanece estable,
    # sin cambios -- el detalle vive aparte, en evidence_length_violations.
    assert validate_judge_result(result) == ["argumentative_clarity:evidence_item_over_20_words"]


@scenario("J08-03. El retry recibe feedback específico (ubicación+conteo exactos, nunca solo el código genérico) y una segunda respuesta <=20 -> válido")
def test_j08_03_retry_receives_specific_feedback_and_recovers():
    overlength = _full_result(LONG_EVIDENCE)
    actual_count = len(LONG_EVIDENCE[0].split())
    llm, out = _run_judge([json.dumps(overlength), json.dumps(_full_result(SHORT_EVIDENCE))])

    assert llm.calls == 2
    second_prompt = llm.prompts[1]
    assert "argumentative_clarity" in second_prompt
    assert "evidence_from_generated[0]" in second_prompt
    assert f"{actual_count} palabras" in second_prompt
    assert "máximo permitido = 20" in second_prompt
    # Nunca se le pide truncar -- se le pide reformular.
    assert "reformula" in second_prompt.lower()
    assert validate_judge_result(out["result"]) == []


@scenario("J08-04. score y justification de la respuesta CORREGIDA (segundo intento) se preservan exactamente -- nunca alterados por el mecanismo de retry")
def test_j08_04_score_and_justification_preserved():
    overlength = _full_result(LONG_EVIDENCE)
    corrected = _full_result(SHORT_EVIDENCE)
    llm, out = _run_judge([json.dumps(overlength), json.dumps(corrected)])

    expected = corrected["scores"]["argumentative_clarity"]
    actual = out["result"]["scores"]["argumentative_clarity"]
    assert actual["score"] == expected["score"]
    assert actual["justification"] == expected["justification"]
    assert actual["evidence_from_generated"] == expected["evidence_from_generated"]


@scenario("J08-05. Las 5 dimensiones de JUDGE_CRITERIA usan exactamente la misma regla de 20 palabras -- ninguna excepción por dimensión")
def test_j08_05_all_five_dimensions_same_rule():
    all_long = {
        "scores": {criterion: _score_item(evidence=LONG_EVIDENCE) for criterion in JUDGE_CRITERIA},
        "strengths": [], "organization_differences": [], "missing_topics_or_omissions": [],
        "overall_assessment": "Evaluación general.",
    }
    errors = validate_judge_result(all_long)
    length_errors = sorted(e for e in errors if e.endswith(":evidence_item_over_20_words"))
    assert length_errors == sorted(f"{criterion}:evidence_item_over_20_words" for criterion in JUDGE_CRITERIA)

    violations = evidence_length_violations(all_long)
    assert len(violations) == len(JUDGE_CRITERIA)
    assert {v["criterion"] for v in violations} == set(JUDGE_CRITERIA)
    assert all(v["max_word_count"] == EVIDENCE_ITEM_MAX_WORDS for v in violations)


@scenario("J08-06. Error de longitud + otro error no relacionado (score inválido) en la misma respuesta -> retry normal, nunca repair/aceptación silenciosa")
def test_j08_06_mixed_errors_never_silently_accepted():
    mixed = _full_result(LONG_EVIDENCE, overrides={"coherence": {"score": 99}})
    llm, out = _run_judge([json.dumps(mixed), json.dumps(_full_result(SHORT_EVIDENCE))])

    assert llm.calls == 2  # nunca se acepta el intento 1 -- tiene que reintentar
    first_attempt_errors = out["attempt_errors"][0]
    assert any("score_not_integer_1_to_5" in e for e in first_attempt_errors)
    assert any(e.endswith("evidence_item_over_20_words") for e in first_attempt_errors)
    # El error NO relacionado con longitud se preserva tal cual en el
    # feedback del siguiente prompt (error_code estable, sin enriquecer).
    feedback = build_retry_feedback(mixed, first_attempt_errors)
    assert any("score_not_integer_1_to_5" in f for f in feedback)
    assert validate_judge_result(out["result"]) == []


@scenario("J08-07. Tres incumplimientos consecutivos (max_attempts=3) -> ValueError fail-closed, nunca una sección/resultado falsamente válido")
def test_j08_07_three_failures_raise_fail_closed():
    overlength = _full_result(LONG_EVIDENCE)
    llm = FakeJudgeLLM([json.dumps(overlength)] * 3)
    raised = False
    try:
        run_llm_judge(
            topic_name="t", source_stage="06_agente_redactor", automatic_metrics={}, factual_metrics={},
            generated_plain_text="x " * 50, ground_truth_plain_text="y " * 50,
            max_generated_chars=4000, max_ground_truth_chars=4000, max_attempts=3,
            llm_factory=lambda: llm,
        )
    except ValueError:
        raised = True
    assert raised
    assert llm.calls == 3


@scenario("J08-08. PARTIAL_HALT (evaluation_pipeline_outcome.py, routing 07->08) permanece completamente intacto -- este cambio no lo toca")
def test_j08_08_partial_halt_routing_untouched():
    import inspect

    from src.adapters import evaluation_pipeline_outcome

    source = inspect.getsource(evaluation_pipeline_outcome)
    assert "SCIENTIFIC_EXHAUSTION_HALT_REASON_CODES" in source
    assert "allow_partial_halt" in source
    # Este módulo no importa nada de llm_judge.py -- confirma que el
    # cambio de feedback estructurado no puede afectar el routing.
    assert "llm_judge" not in source


@scenario("J08-09. validate_judge_result usa EVIDENCE_ITEM_MAX_WORDS como fuente única de verdad -- no un literal 20 duplicado")
def test_j08_09_validate_judge_result_uses_constant_as_source_of_truth():
    import inspect

    from src.tools.evaluation import llm_judge as module

    source = inspect.getsource(module.validate_judge_result)
    assert "EVIDENCE_ITEM_MAX_WORDS" in source
    assert "> 20" not in source  # el literal ya no aparece -- solo la constante

    # Prueba de comportamiento real, no solo textual: cambiar el límite
    # (monkeypatch de la constante) debe cambiar el resultado de
    # validate_judge_result para el MISMO evidence_item -- confirma que
    # la función REALMENTE lee la constante en tiempo de ejecución, no
    # un valor congelado en otro lugar.
    text_21_words = "una " * 21
    result = _full_result([text_21_words.strip()])

    original = module.EVIDENCE_ITEM_MAX_WORDS
    try:
        assert module.validate_judge_result(result) == ["argumentative_clarity:evidence_item_over_20_words"]
        module.EVIDENCE_ITEM_MAX_WORDS = 25
        assert module.validate_judge_result(result) == []
    finally:
        module.EVIDENCE_ITEM_MAX_WORDS = original

    # El error_code histórico permanece exactamente igual, para no
    # romper consumidores existentes que dependan de ese string exacto.
    assert module.EVIDENCE_ITEM_MAX_WORDS == 20
    assert module.validate_judge_result(result) == ["argumentative_clarity:evidence_item_over_20_words"]


if __name__ == "__main__":
    for fn in (
        test_j08_01_short_evidence_valid_single_call,
        test_j08_02_error_identifies_criterion_index_count,
        test_j08_03_retry_receives_specific_feedback_and_recovers,
        test_j08_04_score_and_justification_preserved,
        test_j08_05_all_five_dimensions_same_rule,
        test_j08_06_mixed_errors_never_silently_accepted,
        test_j08_07_three_failures_raise_fail_closed,
        test_j08_08_partial_halt_routing_untouched,
        test_j08_09_validate_judge_result_uses_constant_as_source_of_truth,
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
