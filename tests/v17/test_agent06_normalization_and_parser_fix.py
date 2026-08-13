"""Bug real reproducible en Exp07, ``05_generador_esquema`` -> ``06_
agente_redactor``: S3 registró ``EMPTY_DRAFT_TEXT``/``word_count=0``
aunque el archivo crudo real (``S3_attempt_3.txt``) contenía un JSON
válido, envuelto en fence Markdown, con 1775 caracteres de
``draft_text`` y 6 claims.

Causa raíz confirmada leyendo el código real y reproduciendo contra el
archivo exacto del experimento: el parser JSON (``DraftWritingRuntime.
parse``) YA manejaba el fence correctamente -- el bug real estaba en
``normalize_generated_section`` (``src/tools/draft_writing/
normalization.py``): cuando ninguna oración del ``draft_text`` tenía
citas inline y NINGUNA coincidía textualmente (exacto, sin fuzzy
matching) con algún ``claims[].claim``, la oración se BORRABA por
completo (``continue``) -- si todas las oraciones sustantivas caían en
ese caso, ``draft_text`` quedaba vacío, produciendo ``EMPTY_DRAFT_
TEXT`` en vez del motivo real (``uncited_substantive_sentence`` /
``missing_claim_for_sentence``).

Fix, fail-closed, sin heurísticas:
- Citas inline válidas o correspondencia EXACTA (nunca fuzzy/semántica)
  con un claim declarado: se heredan normalmente.
- Sin esa correspondencia: la oración se PRESERVA en draft_text, sin
  cita inventada y sin claim_entry asociado -- nunca se borra, nunca
  colapsa la sección a texto vacío. La validación posterior reporta la
  causa real.

También se corrigió, por separado: ``DraftWritingRuntime.parse`` ahora
reutiliza ``parse_json_safely`` (ya probado en producción,
``src/tools/evaluation/llm_judge.py``) en vez de duplicar lógica de
extracción de JSON; y ``raw_section_outputs`` ahora versiona por
``agent_attempt_{NN}/`` (``agent_input.attempt_number``) para no
sobrescribir en silencio los artefactos de un intento externo anterior
tras un RETRY/HALT_STAGE."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "v16"))

from test_agent06_v16 import Env  # noqa: E402

from src.adapters.draft_writing_runtime import DraftWritingRuntime  # noqa: E402
from src.agents.draft_writing_agent import DraftWritingAgent  # noqa: E402
from src.tools.draft_writing.normalization import normalize_generated_section  # noqa: E402
from src.tools.draft_writing.validation import validate_generated_section  # noqa: E402

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


_RUNTIME = DraftWritingRuntime(invoke_fn=lambda p: None, collection=None)


@scenario("GG01. Parser: JSON puro se parsea correctamente")
def test_pure_json():
    result = _RUNTIME.parse('{"section_id": "S1", "draft_text": "x", "claims": []}')
    assert result == {"section_id": "S1", "draft_text": "x", "claims": []}


@scenario("GG02. Parser: JSON envuelto en fence Markdown ```json ... ``` se recupera correctamente")
def test_fenced_json():
    raw = '```json\n{"section_id": "S1", "draft_text": "x", "claims": []}\n```'
    result = _RUNTIME.parse(raw)
    assert result == {"section_id": "S1", "draft_text": "x", "claims": []}

    # Fence sin la etiqueta "json" -- también debe recuperarse.
    raw_plain_fence = '```\n{"section_id": "S1", "draft_text": "y", "claims": []}\n```'
    result2 = _RUNTIME.parse(raw_plain_fence)
    assert result2 == {"section_id": "S1", "draft_text": "y", "claims": []}


@scenario("GG03. Parser: entrada inválida produce error explícito, nunca un dict vacío ni draft_text=''")
def test_invalid_json_fails_closed():
    try:
        _RUNTIME.parse("esto no contiene ningún JSON válido en absoluto")
    except ValueError as exc:
        assert "INVALID_LLM_OUTPUT" in str(exc)
    else:
        raise AssertionError("una entrada sin JSON válido debe fallar cerrado")


@scenario("GG04. Oración y claim EXACTAMENTE compatibles -> hereda la cita del claim correctamente, comportamiento normal preservado")
def test_exact_sentence_claim_match_inherits_citation():
    section = {
        "section_id": "S1",
        "draft_text": "El modelo alcanza una precisión del noventa por ciento en el conjunto de prueba.",
        "claims": [{
            "claim": "El modelo alcanza una precisión del noventa por ciento en el conjunto de prueba",
            "supporting_citations": ["[a.pdf | c1]"],
        }],
    }
    allowed = {("a.pdf", "c1")}
    result = normalize_generated_section(section, allowed)
    assert "[a.pdf | c1]" in result["draft_text"]
    assert len(result["claims"]) == 1
    assert result["claims"][0]["supporting_citations"] == ["[a.pdf | c1]"]


@scenario("GG05. Oración sin cita y claim parafraseado (no coincide textualmente) -> la oración NO desaparece, NO recibe cita inventada, validación falla con el motivo real")
def test_paraphrased_claim_never_drops_sentence_or_invents_citation():
    section = {
        "section_id": "S1",
        "draft_text": "El crecimiento poblacional impulsa el aumento de la huella ecológica en la región estudiada.",
        "claims": [{
            # Paráfrasis real -- mismo contenido, texto distinto, tal
            # como ocurrió en el caso real de Exp07.
            "claim": "La urbanización y el crecimiento demográfico son factores que incrementan la huella ecológica regional",
            "supporting_citations": ["[b.pdf | c9]"],
        }],
    }
    allowed = {("b.pdf", "c9")}
    result = normalize_generated_section(section, allowed)

    # La oración se preserva, textualmente, sin cita inventada.
    assert "crecimiento poblacional" in result["draft_text"]
    assert "[b.pdf | c9]" not in result["draft_text"]
    assert result["draft_text"].strip() != ""
    # Nunca se crea un claim_entry para una oración sin correspondencia segura.
    assert result["claims"] == []

    validation = validate_generated_section(
        result, {"section_id": "S1", "section_title": "t", "section_type": "fundamentos"}, evidence=[],
    )
    assert validation["validation_ok"] is False
    assert "EMPTY_DRAFT_TEXT" not in (validation.get("errors") or [])
    assert (
        "uncited_substantive_sentence" in (validation.get("citation_errors") or [])
        or "missing_claim_for_sentence" in (validation.get("claim_errors") or [])
    )


@scenario("GG06. Varias oraciones sustantivas SIN correspondencia -- nunca produce EMPTY_DRAFT_TEXT simplemente porque la normalización no pudo asociarlas a claims (reproducción del caso real de Exp07)")
def test_multiple_unmatched_sentences_never_produce_empty_draft_text():
    section = {
        "section_id": "S3",
        "draft_text": (
            "El crecimiento poblacional y la urbanización impulsan la huella ecológica regional. "
            "La actividad económica intensifica el uso de recursos naturales disponibles en el territorio. "
            "Los conflictos de uso del suelo agravan la degradación ambiental observada en la zona."
        ),
        "claims": [
            {"claim": "Afirmación totalmente distinta sin relación textual alguna con el cuerpo del texto", "supporting_citations": ["[x.pdf | c1]"]},
        ],
    }
    allowed = {("x.pdf", "c1")}
    result = normalize_generated_section(section, allowed)

    assert result["draft_text"].strip() != ""
    assert len(result["draft_text"]) > 100  # las 3 oraciones siguen presentes

    validation = validate_generated_section(
        result, {"section_id": "S3", "section_title": "t", "section_type": "fundamentos"}, evidence=[],
    )
    assert validation["validation_ok"] is False
    all_errors = (
        list(validation.get("errors") or [])
        + list(validation.get("citation_errors") or [])
        + list(validation.get("claim_errors") or [])
    )
    assert "EMPTY_DRAFT_TEXT" not in all_errors


@scenario("GG07. Dos intentos EXTERNOS de Agent06 (RETRY tras HALT) sobre el MISMO output_directory -- los artefactos raw de AMBOS quedan preservados, ninguno sobrescribe al otro")
def test_two_external_attempts_preserve_both_raw_artifacts():
    from dataclasses import replace

    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(
        lambda p: json.dumps({
            "section_id": "S1", "section_title": "Methods",
            "draft_text": "This unsupported scientific statement lacks any valid evidence citation.",
            "claims": [],
        }),
        e.collection,
    ))
    result1 = e.agent.execute(e.ai)
    assert result1.requested_transition.action.value in ("RETRY", "HALT_STAGE")
    attempt1_dir = e.out / "raw_section_outputs" / "agent_attempt_01"
    assert attempt1_dir.is_dir()
    attempt1_files = sorted(p.name for p in attempt1_dir.glob("S1_attempt_*.txt"))
    assert attempt1_files

    # Segundo intento EXTERNO, MISMO output_directory real -- simula la
    # reanudación real del orquestador tras el RETRY (solo cambia
    # attempt_number y previous_attempt, igual que haría el pipeline real).
    from src.contracts.agent_input import PreviousAttemptSummary

    ai2 = replace(
        e.ai, attempt_number=2,
        previous_attempt=PreviousAttemptSummary(quality_status="NEEDS_REVISION"),
    )
    e.agent = DraftWritingAgent(DraftWritingRuntime(
        lambda p: json.dumps({
            "section_id": "S1", "section_title": "Methods",
            "draft_text": "This unsupported scientific statement lacks any valid evidence citation.",
            "claims": [],
        }),
        e.collection,
    ))
    result2 = e.agent.execute(ai2)
    assert result2.requested_transition.action.value in ("RETRY", "HALT_STAGE")

    attempt1_dir_after = e.out / "raw_section_outputs" / "agent_attempt_01"
    attempt2_dir = e.out / "raw_section_outputs" / "agent_attempt_02"
    assert attempt1_dir_after.is_dir()
    # El intento 1 sigue exactamente igual -- ningún archivo se sobrescribió.
    assert sorted(p.name for p in attempt1_dir_after.glob("S1_attempt_*.txt")) == attempt1_files
    assert attempt2_dir.is_dir()
    assert list(attempt2_dir.glob("S1_attempt_*.txt"))


@scenario("GG08. Contrato raw_section_outputs: el ArtifactReference SIEMPRE apunta al directorio padre (histórico completo), nunca a un subdirectorio de intento")
def test_raw_section_outputs_artifact_points_to_parent_directory():
    e = Env(attempt=1)
    e.agent = DraftWritingAgent(DraftWritingRuntime(
        lambda p: json.dumps({
            "section_id": "S1", "section_title": "Methods",
            "draft_text": "This unsupported scientific statement lacks any valid evidence citation.",
            "claims": [],
        }),
        e.collection,
    ))
    result = e.agent.execute(e.ai)
    ref = result.output_artifacts["raw_section_outputs"]
    assert Path(ref.path) == e.out / "raw_section_outputs"
    assert Path(ref.path).name == "raw_section_outputs"
    # El subdirectorio del intento específico se registra aparte, como
    # metadata explícita -- nunca sustituye al ArtifactReference padre.
    report = json.loads((e.out / "draft_validation_report.json").read_text())
    assert report["current_raw_attempt_directory"] == str(e.out / "raw_section_outputs" / "agent_attempt_01")


@scenario("GG09. Fingerprint: normalization_version participa en current_fingerprint -- un cambio de contrato de normalización invalida el fingerprint sin --force-rerun")
def test_normalization_version_participates_in_fingerprint():
    from src.adapters.draft_writing_runtime import _draft_signature
    from src.state.fingerprints import fingerprint_mapping

    base_policy = {
        "stage_version": "S", "prompt_version": "P", "rag_version": "R", "validation_version": "V",
        "normalization_version": "sentence_claim_exact_match_preserve_unmatched_v1",
    }
    cfg_common = {"experiment_id": "e", "experiment_dir": "d", "model": "m", "embedding_model_name": "em", "chroma_collection_name": "c"}
    fp_with_version = fingerprint_mapping(_draft_signature({**cfg_common, "policy": base_policy}, {}))
    older_policy = {k: v for k, v in base_policy.items() if k != "normalization_version"}
    fp_without_version = fingerprint_mapping(_draft_signature({**cfg_common, "policy": older_policy}, {}))
    assert fp_with_version != fp_without_version

    # Cambiar el VALOR de normalization_version (una futura corrección
    # de contrato) también debe cambiar el fingerprint.
    bumped_policy = {**base_policy, "normalization_version": "otra_version_futura_v2"}
    fp_bumped = fingerprint_mapping(_draft_signature({**cfg_common, "policy": bumped_policy}, {}))
    assert fp_bumped != fp_with_version


@scenario("GG10. Utilidad JSON: parse_json_safely vive en src.utils.json_parsing (neutral) -- llm_judge.py y draft_writing_runtime.py usan literalmente la MISMA función, sin duplicación")
def test_parse_json_safely_shared_across_domains_without_duplication():
    from src.utils.json_parsing import parse_json_safely as from_utils
    from src.tools.evaluation.llm_judge import parse_json_safely as from_judge
    import src.adapters.draft_writing_runtime as dwr_module
    import inspect

    assert from_judge is from_utils  # reexportado, no copiado
    # draft_writing_runtime.py no debe importar nada de
    # src.tools.evaluation -- verificado por inspección de su propio
    # código fuente, no solo de sus imports en tiempo de ejecución.
    source = inspect.getsource(dwr_module)
    assert "src.tools.evaluation" not in source


if __name__ == "__main__":
    for fn in (
        test_pure_json,
        test_fenced_json,
        test_invalid_json_fails_closed,
        test_exact_sentence_claim_match_inherits_citation,
        test_paraphrased_claim_never_drops_sentence_or_invents_citation,
        test_multiple_unmatched_sentences_never_produce_empty_draft_text,
        test_two_external_attempts_preserve_both_raw_artifacts,
        test_raw_section_outputs_artifact_points_to_parent_directory,
        test_normalization_version_participates_in_fingerprint,
        test_parse_json_safely_shared_across_domains_without_duplication,
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
