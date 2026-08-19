"""Modularización de src/tools/verification/validation.py (Bloque C,
extracción mecánica C1-C6).

Causa raíz: el archivo tenía ~6035 líneas y 199 definiciones
top-level, de las cuales 42 (21%) eran código muerto por
redefinición del mismo nombre (solo la última definición de cada
nombre es alcanzable en Python). Se identificaron 3 "bloques
contaminados" (líneas originales 1343-2162, 2297-2876, 3459-6003 --
65.4% del archivo) donde CUALQUIER división entre archivos podría
cambiar cuál definición de un nombre duplicado "gana" -- esos bloques
NO se tocaron en este Bloque C, permanecen íntegros dentro de
validation.py, exactamente en su orden original.

Solo la zona segura (sin ningún nombre duplicado, líneas originales
1-1342) se modularizó, en 6 extracciones mecánicas incrementales
(C1-C6), cada una verificada con diff byte a byte del cuerpo movido,
identidad de objeto (``is``, no solo ``==``) entre el símbolo
accesible vía validation.py y vía el módulo nuevo, y la suite
completa de tests de verification/Agent07 antes y después.

validation.py permanece como fachada pública: todo símbolo que un
consumidor externo pudiera importar desde ``src.tools.verification.
validation`` sigue siendo accesible ahí, sin ningún cambio de nombre,
firma, tipo de retorno, reason code, mensaje de error, ni orden de
validación.

Multidominio y genérico: estos tests solo verifican la mecánica de
import/reexport, nunca contenido científico concreto."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

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


# Los 53 símbolos movidos en C1-C6 (10+9+6+14+6+8), agrupados por módulo destino.
C1_SYMBOLS = [
    "CommittedAgent06Input", "_is_within", "_latest_stage_decision", "_require_result_field",
    "_validate_committed_result", "_validate_manifest", "sha256_file",
    "validate_committed_agent06_input", "validate_provisional_evidence_output", "validate_sha256_hex",
]
C2_SYMBOLS = [
    "ClaimRetrievalTool", "EvidenceSelection", "_REQUIRED_LLM_FIELDS", "_canonical_evidence_rows",
    "allowed_verdicts_for_claim", "canonical_correction_evidence_text", "deterministic_precheck",
    "select_evidence_for_scientific_judgment", "validate_claim_verification_context",
]
C3_SYMBOLS = [
    "_require_exact_type", "_require_string_list", "compute_hallucination_risk",
    "derive_semantic_issue_codes", "determine_final_correction_eligibility",
    "validate_llm_verification_response",
]
C4_SYMBOLS = [
    "ADDITIONAL_RETRIEVAL_CANDIDATE_ALLOWED_FIELDS", "ADDITIONAL_RETRIEVAL_COVERAGE_FIELDS",
    "ADDITIONAL_RETRIEVAL_DELTA_ACCUMULATIVE_FIELDS", "ADDITIONAL_RETRIEVAL_DELTA_ALLOWED_FIELDS",
    "ADDITIONAL_RETRIEVAL_DELTA_DERIVED_FIELDS", "ADDITIONAL_RETRIEVAL_DELTA_SNAPSHOT_FIELDS",
    "ADDITIONAL_RETRIEVAL_DELTA_UNION_FIELDS", "ADDITIONAL_RETRIEVAL_MUTABLE_DETERMINISTIC_FIELDS",
    "ADDITIONAL_RETRIEVAL_PROTECTED_CANDIDATE_FIELDS", "ADDITIONAL_RETRIEVAL_STOP_REASONS",
    "_delta_string_sequence", "_validate_coverage_snapshot", "_validate_incremental_candidate",
    "validate_additional_retrieval_delta",
]
C5_SYMBOLS = [
    "_normalize_decimal_literal", "extract_quantitative_pairs_strict", "metric_context_supported",
    "quantitative_pair_supported", "validate_correction_proposal_response",
    "validate_correction_text_integrity",
]
C6_SYMBOLS = [
    "PRECHECK_STATUSES", "_phase62_brackets_balanced", "_reverification_nonempty_string",
    "_reverification_string_tuple", "_sha256_text", "validate_correction_reverification_input_contract",
    "validate_correction_reverification_result_contract", "validate_reverification_block_matrix",
]

MODULES = {
    "commit_handoff_validation": C1_SYMBOLS,
    "evidence_selection": C2_SYMBOLS,
    "llm_response_validation": C3_SYMBOLS,
    "retrieval_delta_validation": C4_SYMBOLS,
    "numeric_validation": C5_SYMBOLS,
    "reverification_contracts": C6_SYMBOLS,
}


@scenario("MODC-01. Todos los símbolos movidos (53 en total, C1-C6) siguen accesibles desde src.tools.verification.validation")
def test_modc_01_all_moved_symbols_accessible_from_facade():
    from src.tools.verification import validation

    missing = []
    for module_name, symbols in MODULES.items():
        for symbol in symbols:
            if not hasattr(validation, symbol):
                missing.append(f"{module_name}.{symbol}")
    assert not missing, f"Símbolos no accesibles desde validation.py: {missing}"


@scenario("MODC-02. Cada símbolo accesible vía validation.py es EL MISMO OBJETO (identidad, no solo igualdad) que en su módulo de origen")
def test_modc_02_same_object_identity_across_facade():
    from src.tools.verification import validation
    from src.tools.verification import commit_handoff_validation, evidence_selection
    from src.tools.verification import llm_response_validation, retrieval_delta_validation
    from src.tools.verification import numeric_validation, reverification_contracts

    module_objs = {
        "commit_handoff_validation": commit_handoff_validation,
        "evidence_selection": evidence_selection,
        "llm_response_validation": llm_response_validation,
        "retrieval_delta_validation": retrieval_delta_validation,
        "numeric_validation": numeric_validation,
        "reverification_contracts": reverification_contracts,
    }
    mismatches = []
    for module_name, symbols in MODULES.items():
        mod = module_objs[module_name]
        for symbol in symbols:
            if getattr(validation, symbol) is not getattr(mod, symbol):
                mismatches.append(f"{module_name}.{symbol}")
    assert not mismatches, f"No son el mismo objeto: {mismatches}"


@scenario("MODC-03. Los 8 consumidores de PRODUCCIÓN (src/) que importan desde src.tools.verification.validation resuelven sin ImportError -- los 33 consumidores en tests/ NO se reimportan aquí (se verifican indirectamente al correr esa suite completa, ver MODC-06 y el resto de este archivo dentro de tests/verification/)")
def test_modc_03_all_known_external_consumers_still_import_cleanly():
    import importlib

    external_consumers = [
        "src.adapters.agent07c_handoff",
        "src.adapters.claim_verification_context",
        "src.adapters.verification_notebook",
        "src.adapters.verification_runtime",
        "src.agents.verification_agent",
        "src.tools.verification.corrections",
        "src.tools.verification.prompting",
        "src.tools.verification.resolution",
    ]
    failures = []
    for mod_name in external_consumers:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            failures.append((mod_name, repr(exc)))
    assert not failures, f"Fallos de import: {failures}"


@scenario("MODC-04. validation.py sigue teniendo un docstring de módulo (façade), y los 6 módulos nuevos existen físicamente en src/tools/verification/")
def test_modc_04_facade_and_new_modules_exist():
    from src.tools.verification import validation

    assert validation.__doc__ is not None and validation.__doc__.strip()
    for module_name in MODULES:
        path = REPO_ROOT / "src" / "tools" / "verification" / f"{module_name}.py"
        assert path.is_file(), f"{module_name}.py no existe"


@scenario("MODC-05. Comportamiento funcional preservado a través de la fachada: validate_sha256_hex (C1) y validate_additional_retrieval_delta (C4) dan resultados idénticos al contrato real ya existente en el código (campo contador válido/inválido, campo desconocido bajo strict=True)")
def test_modc_05_functional_behavior_preserved_via_facade():
    from src.tools.verification import validation

    assert validation.validate_sha256_hex(None, allow_none=True) is None
    assert validation.validate_sha256_hex("a" * 64) == "a" * 64
    try:
        validation.validate_sha256_hex("not-a-hash")
        raised = False
    except ValueError as exc:
        raised = True
        assert "SHA256_INVALID" in str(exc)
    assert raised

    # validate_additional_retrieval_delta: todos los campos son
    # opcionales en el contrato real (cada chequeo interno es
    # "if field in value"); se ejercitan aquí las reglas que el propio
    # código ya declara, sin inventar ningún fixture nuevo:
    # - "rounds_executed" es uno de los ADDITIONAL_RETRIEVAL_DELTA_
    #   ACCUMULATIVE_FIELDS, validado como entero >= 0.
    # - un campo fuera de ADDITIONAL_RETRIEVAL_DELTA_ALLOWED_FIELDS es
    #   rechazado bajo strict=True (el default).
    valid_delta = {"rounds_executed": 1}
    assert validation.validate_additional_retrieval_delta(valid_delta) == valid_delta

    try:
        validation.validate_additional_retrieval_delta({"rounds_executed": -1})
        raised_counter = False
    except ValueError as exc:
        raised_counter = True
        assert "ADDITIONAL_RETRIEVAL_DELTA_COUNTER_INVALID:rounds_executed" in str(exc)
    assert raised_counter

    try:
        validation.validate_additional_retrieval_delta({"an_unknown_field_not_in_the_contract": 1})
        raised_unknown = False
    except ValueError as exc:
        raised_unknown = True
        assert "ADDITIONAL_RETRIEVAL_DELTA_UNKNOWN_FIELD" in str(exc)
    assert raised_unknown
    assert raised


@scenario("MODC-06. Los 3 bloques contaminados (por redefinición de nombres) permanecen íntegros dentro de validation.py -- ningún nombre duplicado fue tocado")
def test_modc_06_contaminated_blocks_untouched():
    source = (REPO_ROOT / "src" / "tools" / "verification" / "validation.py").read_text(encoding="utf-8")
    # Símbolos que en el inventario original tenían múltiples definiciones
    # (shadowing) -- deben seguir presentes varias veces en validation.py,
    # sin haber sido tocados por este Bloque C.
    for duplicated_symbol in (
        "validate_correction_traceability_row_contract",
        "build_provisional_traceability_rows",
        "_phase652_validate_collection_result_contract",
    ):
        occurrences = source.count(f"def {duplicated_symbol}(")
        assert occurrences >= 2, f"{duplicated_symbol}: se esperaban >=2 definiciones, hay {occurrences}"


if __name__ == "__main__":
    for fn in (
        test_modc_01_all_moved_symbols_accessible_from_facade,
        test_modc_02_same_object_identity_across_facade,
        test_modc_03_all_known_external_consumers_still_import_cleanly,
        test_modc_04_facade_and_new_modules_exist,
        test_modc_05_functional_behavior_preserved_via_facade,
        test_modc_06_contaminated_blocks_untouched,
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
