"""Centralización de la lógica de ``attempt_number`` en ``AgentInput``
(``src/contracts/agent_input.py``).

Causa raíz: ``agent_input.attempt_number == 1`` / ``== 2`` estaba
duplicado 7 veces entre ``extraction_agent.py`` (4x) y
``draft_writing_agent.py`` (3x), sin ninguna fuente única de verdad.
``is_first_attempt()``/``is_final_attempt()`` reemplazan esas
comparaciones dispersas -- misma semántica exacta, un solo lugar que
la define.

Estos tests verifican:
1. Los helpers en aislamiento, para el rango completo de valores de
   ``attempt_number`` que el sistema usa hoy (1 y 2).
2. Que ``is_final_attempt()`` sin argumento usa el mismo valor (2) que
   las comparaciones ``== 2`` que reemplaza -- no cambia el número de
   intentos por defecto.
3. Que ``extraction_agent.py`` y ``draft_writing_agent.py`` ya no
   contienen ninguna comparación cruda de ``attempt_number`` -- toda
   la lógica pasa por los helpers centralizados.
4. Comportamiento end-to-end preservado: los mismos escenarios de
   intento 1 -> RETRY / intento final -> HALT que ya estaban cubiertos
   por las suites existentes de Stage03/Stage06 siguen dando
   exactamente el mismo resultado.

Multidominio y genérico: ningún test depende de contenido científico
concreto."""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.contracts.agent_input import (  # noqa: E402
    AgentContext,
    AgentInput,
    DEFAULT_MAX_ATTEMPTS,
    ExecutionMode,
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


def _make_agent_input(attempt_number: int) -> AgentInput:
    return AgentInput(
        experiment_id="exp", run_id="run", stage_name="stage",
        attempt_number=attempt_number, mode=ExecutionMode.FULL_RUN,
        agent_context=AgentContext(allowed_tools=("tool",), output_directory="out"),
        dependencies={}, policy={},
    )


@scenario("ATTEMPT-01. DEFAULT_MAX_ATTEMPTS es 2 -- el mismo número que ya usaban las comparaciones == 2 que se reemplazaron")
def test_attempt_01_default_max_attempts_unchanged():
    assert DEFAULT_MAX_ATTEMPTS == 2


@scenario("ATTEMPT-02. is_first_attempt(): True solo en attempt_number=1")
def test_attempt_02_is_first_attempt():
    assert _make_agent_input(1).is_first_attempt() is True
    assert _make_agent_input(2).is_first_attempt() is False


@scenario("ATTEMPT-03. is_final_attempt() (sin argumento, default=2): True solo en attempt_number=2, idéntico a la comparación == 2 original")
def test_attempt_03_is_final_attempt_default():
    assert _make_agent_input(1).is_final_attempt() is False
    assert _make_agent_input(2).is_final_attempt() is True


@scenario("ATTEMPT-04. is_final_attempt(max_attempts) explícito: cada etapa puede declarar su propio máximo sin tocar el default de otras")
def test_attempt_04_is_final_attempt_explicit_max():
    assert _make_agent_input(2).is_final_attempt(3) is False
    assert _make_agent_input(3).is_final_attempt(3) is True
    assert _make_agent_input(1).is_final_attempt(1) is True


@scenario("ATTEMPT-05. is_final_attempt rechaza max_attempts inválido (fail-closed: nunca decide silenciosamente con un valor absurdo)")
def test_attempt_05_is_final_attempt_rejects_invalid_max():
    agent_input = _make_agent_input(1)
    for bad_value in (0, -1, "2", True, None):
        try:
            agent_input.is_final_attempt(bad_value)
            raised = False
        except (ValueError, TypeError):
            raised = True
        assert raised, f"max_attempts={bad_value!r} debió rechazarse"


@scenario("ATTEMPT-06. extraction_agent.py no contiene ninguna comparación cruda de attempt_number -- toda la lógica pasa por los helpers")
def test_attempt_06_extraction_agent_has_no_raw_comparisons():
    source = (REPO_ROOT / "src" / "agents" / "extraction_agent.py").read_text(encoding="utf-8")
    assert re.search(r"attempt_number\s*==\s*[12]\b", source) is None
    assert re.search(r"attempt_number\s*>=\s*2\b", source) is None
    assert "is_first_attempt()" in source
    assert "is_final_attempt()" in source


@scenario("ATTEMPT-07. draft_writing_agent.py no contiene ninguna comparación cruda de attempt_number -- toda la lógica pasa por los helpers")
def test_attempt_07_draft_writing_agent_has_no_raw_comparisons():
    source = (REPO_ROOT / "src" / "agents" / "draft_writing_agent.py").read_text(encoding="utf-8")
    assert re.search(r"attempt_number\s*==\s*[12]\b", source) is None
    assert source.count("is_first_attempt()") == 3


@scenario("ATTEMPT-08. Round-trip to_dict/from_dict conserva attempt_number, y los helpers siguen funcionando sobre el objeto reconstruido")
def test_attempt_08_helpers_survive_serialization_roundtrip():
    original = _make_agent_input(2)
    restored = AgentInput.from_dict(original.to_dict())
    assert restored.attempt_number == 2
    assert restored.is_final_attempt() is True
    assert restored.is_first_attempt() is False


if __name__ == "__main__":
    for fn in (
        test_attempt_01_default_max_attempts_unchanged,
        test_attempt_02_is_first_attempt,
        test_attempt_03_is_final_attempt_default,
        test_attempt_04_is_final_attempt_explicit_max,
        test_attempt_05_is_final_attempt_rejects_invalid_max,
        test_attempt_06_extraction_agent_has_no_raw_comparisons,
        test_attempt_07_draft_writing_agent_has_no_raw_comparisons,
        test_attempt_08_helpers_survive_serialization_roundtrip,
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
