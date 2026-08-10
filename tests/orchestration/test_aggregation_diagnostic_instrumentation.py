"""Instrumentación de diagnóstico TEMPORAL para capturar el registro
crudo y el mensaje real de excepción de un elemento inválido durante la
agregación de 07 -- solicitada para investigar
``AGGREGATION_COLLECTION_ELEMENT_INVALID:claim_verification_records:13``
en un experimento real, donde el ``ValueError`` real se descarta por
diseño (solo sobrevive ``type(exc).__name__`` en los ``warnings``).

Esta prueba confirma la única propiedad que importa: la instrumentación
es estrictamente ADITIVA -- con el sink activo o inactivo,
``validate_and_normalize_provisional_collections`` produce EXACTAMENTE
el mismo ``aggregation_status``/``collection_validation_status`` y los
mismos ``issue_codes``/``warnings`` -- nunca cambia una decisión
científica, nunca convierte ``INVALID`` en ``PARTIAL`` ni oculta un
registro inválido. La única diferencia observable es un archivo de
diagnóstico adicional (no oficial, no publicado, no leído por ningún
otro punto del pipeline).
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.verification.validation import (  # noqa: E402
    aggregation_diagnostic_sink,
    validate_and_normalize_provisional_collections,
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


def _base_aggregation_input(**overrides):
    base = {
        "claim_verification_records": (),
        "correction_proposals": (), "correction_reverification_inputs": (), "correction_precheck_results": (),
        "independent_reverification_results": (), "before_after_comparison_results": (),
        "policy_versions": {"verification": "v1"},
        "schema_versions": {"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
        "additional_llm_calls": 0, "additional_retrieval_rounds": 0,
        "correction_applied": False, "official_artifacts_created": False,
    }
    base.update(overrides)
    return base


def _invalid_input():
    # Un elemento deliberadamente inválido en claim_verification_records
    # (le falta "claim_verification_result") -- reproduce el mismo tipo
    # de fallo real reportado (AGGREGATION_COLLECTION_ELEMENT_INVALID en
    # esa colección), sin depender de ningún dato productivo real.
    return _base_aggregation_input(claim_verification_records=({"section_id": "S1"},))


@scenario("K01. Sin sink activo (por defecto en todo el resto del código): comportamiento idéntico al de antes de la instrumentación -- ningún archivo se escribe")
def test_no_sink_no_file_no_behavior_change():
    result = validate_and_normalize_provisional_collections(_invalid_input())
    assert result.collection_validation_status == "INVALID"
    assert result.aggregation_status == "INVALID"
    assert result.collection_issue_codes == ("AGGREGATION_COLLECTION_ELEMENT_INVALID:claim_verification_records:0",)
    assert result.collection_warnings == ("AGGREGATION_COLLECTION_ELEMENT_INVALID:claim_verification_records:0:ValueError",)


@scenario("K02. Con sink activo: MISMO aggregation_status/issue_codes/warnings que sin sink -- la instrumentación no cambia ninguna decisión científica")
def test_sink_active_identical_result():
    aggregation_input = _invalid_input()
    result_without_sink = validate_and_normalize_provisional_collections(aggregation_input)

    with tempfile.TemporaryDirectory() as tmp:
        diag_path = Path(tmp) / "decision-id-x" / "aggregation_invalid_elements_debug.json"
        with aggregation_diagnostic_sink(diag_path):
            result_with_sink = validate_and_normalize_provisional_collections(aggregation_input)

        assert result_with_sink.collection_validation_status == result_without_sink.collection_validation_status
        assert result_with_sink.aggregation_status == result_without_sink.aggregation_status
        assert result_with_sink.collection_issue_codes == result_without_sink.collection_issue_codes
        assert result_with_sink.collection_warnings == result_without_sink.collection_warnings
        assert result_with_sink.invalid_element_records == result_without_sink.invalid_element_records
        # No se "arregla" ni se oculta el registro inválido -- sigue INVALID.
        assert result_with_sink.collection_validation_status == "INVALID"


@scenario("K03. El archivo de diagnóstico se escribe SOLO cuando el sink está activo, con exactamente los 7 campos pedidos por elemento inválido")
def test_diagnostic_file_written_with_required_fields():
    with tempfile.TemporaryDirectory() as tmp:
        diag_path = Path(tmp) / "decision-id-y" / "aggregation_invalid_elements_debug.json"
        assert not diag_path.is_file()

        with aggregation_diagnostic_sink(diag_path):
            validate_and_normalize_provisional_collections(_invalid_input())

        assert diag_path.is_file()
        payload = json.loads(diag_path.read_text(encoding="utf-8"))
        assert len(payload["invalid_elements"]) == 1
        record = payload["invalid_elements"][0]
        assert set(record) == {
            "collection", "position", "reason_code", "raw_element_fingerprint",
            "raw_element", "exception_type", "exception_message",
        }
        assert record["collection"] == "claim_verification_records"
        assert record["position"] == 0
        assert record["exception_type"] == "ValueError"
        assert record["raw_element"] == {"section_id": "S1"}
        assert isinstance(record["exception_message"], str) and record["exception_message"]


@scenario("K04. Sin ningún elemento inválido (input válido): con sink activo, no se escribe ningún archivo -- nada que capturar")
def test_no_file_when_nothing_invalid():
    with tempfile.TemporaryDirectory() as tmp:
        diag_path = Path(tmp) / "decision-id-z" / "aggregation_invalid_elements_debug.json"
        with aggregation_diagnostic_sink(diag_path):
            result = validate_and_normalize_provisional_collections(_base_aggregation_input())
        assert result.collection_validation_status == "VALID"
        assert not diag_path.is_file()


@scenario("K05. Fuera del contexto (después de salir de aggregation_diagnostic_sink), el sink vuelve a None -- no queda activado accidentalmente para llamadas posteriores")
def test_sink_resets_after_context():
    with tempfile.TemporaryDirectory() as tmp:
        diag_path = Path(tmp) / "decision-id-w" / "aggregation_invalid_elements_debug.json"
        with aggregation_diagnostic_sink(diag_path):
            validate_and_normalize_provisional_collections(_invalid_input())
        assert diag_path.is_file()

        # Fuera del contexto: otra llamada con el MISMO input inválido no
        # debe escribir en ningún lado (el sink ya se reseteó a None).
        diag_path.unlink()
        validate_and_normalize_provisional_collections(_invalid_input())
        assert not diag_path.is_file()


if __name__ == "__main__":
    for fn in (
        test_no_sink_no_file_no_behavior_change,
        test_sink_active_identical_result,
        test_diagnostic_file_written_with_required_fields,
        test_no_file_when_nothing_invalid,
        test_sink_resets_after_context,
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
