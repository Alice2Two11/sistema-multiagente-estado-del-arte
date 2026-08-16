"""Contrato estructural de Stage 04: el número de secciones deja de
ser una restricción DURA por defecto -- la longitud total en palabras
(validada en etapas posteriores, sin tocar aquí) sigue siendo la
restricción dura del estado del arte.

Gobernado por ``thematic_analysis_policy.structure_policy.
enforce_section_count`` (bool, default ``False``, reutiliza la
abstracción de policy ya existente en ``thematic_analysis_policy_
config.py`` -- ningún mecanismo paralelo):

- ``False`` (nuevo default): ``STRUCTURE_TOO_SHORT``/``STRUCTURE_TOO_
  LONG`` se siguen calculando como diagnóstico (``structure_too_short``/
  ``structure_too_long`` en ``diagnostic_metrics``), pero nunca entran
  a ``failure_reason_codes``, nunca fuerzan ``NEEDS_REVISION``, nunca
  activan ``REGENERATE_STRUCTURE_ONLY``, nunca causan ``HALT_STAGE`` --
  ``validation_ok`` depende solo de errores científicos/estructurales
  reales.
- ``True``: comportamiento histórico exacto, sin cambios.

``min_sections``/``max_sections``/``section_count`` NUNCA se eliminan
-- siguen registrados en diagnostic_metrics (manifest/reporte) como
métricas descriptivas en ambos modos.

Multidominio y genérico: ningún test usa contenido de un experimento
real."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.contracts.agent_input import AgentInput, AgentContext, ArtifactReference, ExecutionMode  # noqa: E402
from src.agents.thematic_analysis_agent import ThematicAnalysisAgent  # noqa: E402
from src.adapters.thematic_analysis_runtime import ThematicRuntimeDependencies, parse_json  # noqa: E402
from src.config.thematic_analysis_policy_config import get_thematic_analysis_policy  # noqa: E402

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


def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


VALID_PAYLOAD = {
    "corpus_summary": {},
    "themes": [{
        "theme_id": "T1", "theme_name": "Modelos", "description": "ANN y SVM",
        "representative_papers": [
            {"source_filename": "a.pdf", "title": "Alpha"},
            {"source_filename": "b.pdf", "title": "Beta"},
        ],
    }],
    "research_gaps": [{
        "gap_id": "G1", "description": "Falta mas evidencia", "basis": "limitations",
        "supporting_sources": ["a.pdf"],
    }],
    # Dos secciones -- deliberadamente por encima de max_sections=1 usado
    # en los escenarios "demasiadas secciones".
    "suggested_state_of_art_structure": [
        {"section_id": "S1", "section_title": "Modelos", "recommended_sources": ["a.pdf", "b.pdf"]},
        {"section_id": "S2", "section_title": "Resultados", "recommended_sources": ["a.pdf", "b.pdf"]},
    ],
    "comparative_dimensions": [{
        "dimension": "Metodo", "description": "Compara modelos", "relevant_sources": ["a.pdf", "b.pdf"],
    }],
}


class ThematicFixture:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output_dir = self.root / "out"
        self.output_dir.mkdir()

        import pandas as pd

        self.kb_csv = self.root / "kb.csv"
        pd.DataFrame([
            {"source_filename": "a.pdf", "title": "Alpha", "include_in_state_of_art": True, "relevance_level": "alta", "methods_or_models": "ANN", "limitations_or_gaps": "Need more data"},
            {"source_filename": "b.pdf", "title": "Beta", "include_in_state_of_art": True, "relevance_level": "alta", "methods_or_models": "SVM", "limitations_or_gaps": "Limited sites"},
        ]).to_csv(self.kb_csv, index=False)
        self.kb_jsonl = self.root / "kb.jsonl"
        self.kb_jsonl.write_text("{}\n", encoding="utf-8")
        self.manifest = self.root / "m.json"
        self.manifest.write_text(json.dumps({
            "experiment_id": "e", "run_id": "r", "stage": "03_agente_extraccion_kb",
            "safety_policy": {"uses_ground_truth": False},
        }), encoding="utf-8")

    def agent_input(self, *, min_sections=1, max_sections=1, enforce_section_count=False, attempt_number=1):
        policy = get_thematic_analysis_policy({
            "manual_review_policy": {"allowed": True},
            "structure_policy": {"enforce_section_count": enforce_section_count},
        })
        policy["min_sections"] = min_sections
        policy["max_sections"] = max_sections
        dependencies = {
            "scientific_knowledge_base_csv": ArtifactReference(str(self.kb_csv), _sha256(self.kb_csv)),
            "scientific_knowledge_base_jsonl": ArtifactReference(str(self.kb_jsonl), _sha256(self.kb_jsonl)),
            "scientific_extraction_manifest": ArtifactReference(str(self.manifest), _sha256(self.manifest)),
        }
        return AgentInput(
            "e", "r", "04_agente_analisis_tematico", attempt_number, ExecutionMode.FULL_RUN,
            AgentContext(("llm",), str(self.output_dir), {}), dependencies, policy, None,
        )

    def agent(self, payload=None):
        p = payload if payload is not None else VALID_PAYLOAD
        return ThematicAnalysisAgent(ThematicRuntimeDependencies(lambda prompt: json.dumps(p), parse_json))


@scenario("SCOUNT-01. enforce=True + demasiadas secciones -> comportamiento histórico bloqueante")
def test_scount_01_enforce_true_too_many_sections_blocks():
    fixture = ThematicFixture()
    result = fixture.agent().execute(fixture.agent_input(min_sections=1, max_sections=1, enforce_section_count=True))
    assert "STRUCTURE_TOO_LONG" in result.failure_reason_codes
    assert result.quality_status.value == "NEEDS_REVISION"
    assert result.requested_transition.action.value == "RETRY"


@scenario("SCOUNT-02. enforce=False + demasiadas secciones -> APPROVED si no hay otros errores")
def test_scount_02_enforce_false_too_many_sections_approved():
    fixture = ThematicFixture()
    result = fixture.agent().execute(fixture.agent_input(min_sections=1, max_sections=1, enforce_section_count=False))
    assert "STRUCTURE_TOO_LONG" not in result.failure_reason_codes
    assert result.quality_status.value == "APPROVED"
    assert result.requested_transition.action.value == "ADVANCE"
    assert result.quality_metrics["scientific"]["structure_too_long"] is True


@scenario("SCOUNT-03. enforce=False + pocas secciones -> no bloquea")
def test_scount_03_enforce_false_too_few_sections_does_not_block():
    fixture = ThematicFixture()
    result = fixture.agent().execute(fixture.agent_input(min_sections=5, max_sections=10, enforce_section_count=False))
    assert "STRUCTURE_TOO_SHORT" not in result.failure_reason_codes
    assert result.quality_status.value == "APPROVED"
    assert result.quality_metrics["scientific"]["structure_too_short"] is True


@scenario("SCOUNT-04. section_count/min_sections/max_sections siguen presentes en auditoría (diagnostic_metrics), en ambos modos")
def test_scount_04_section_metrics_present_in_audit():
    fixture = ThematicFixture()
    for enforce in (True, False):
        result = fixture.agent().execute(fixture.agent_input(min_sections=1, max_sections=1, enforce_section_count=enforce))
        metrics = result.quality_metrics["scientific"]
        assert metrics["section_count"] == 2
        assert metrics["min_sections"] == 1
        assert metrics["max_sections"] == 1
        assert metrics["enforce_section_count"] is enforce

        report_path = result.output_artifacts["validation"].path
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        assert report["metrics"]["section_count"] == 2
        assert report["metrics"]["min_sections"] == 1
        assert report["metrics"]["max_sections"] == 1

        manifest_path = result.output_artifacts["manifest"].path
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert manifest["diagnostic_metrics"]["section_count"] == 2


@scenario("SCOUNT-05. Los límites de longitud en palabras (min/target/max_total_words) no se modifican por este parche")
def test_scount_05_word_length_limits_untouched():
    import inspect

    from src.agents import thematic_analysis_agent as module
    from src.tools.thematic_analysis import coverage_validation as coverage_module

    for source in (inspect.getsource(module), inspect.getsource(coverage_module)):
        assert "min_total_words" not in source
        assert "target_total_words" not in source
        assert "max_total_words" not in source


@scenario("SCOUNT-06. Otras fallas reales de Stage 04 (ej. fuente inventada) siguen bloqueando, independientemente de enforce_section_count")
def test_scount_06_other_real_failures_still_block():
    fixture = ThematicFixture()
    payload = json.loads(json.dumps(VALID_PAYLOAD))
    payload["themes"][0]["representative_papers"] = [{"source_filename": "invented.pdf", "title": "X"}]
    for enforce in (True, False):
        result = fixture.agent(payload).execute(fixture.agent_input(min_sections=1, max_sections=1, enforce_section_count=enforce))
        assert "INVALID_REPRESENTATIVE_SOURCE" in result.failure_reason_codes
        assert result.quality_status.value != "APPROVED"


@scenario("SCOUNT-07. Ground Truth sigue sin utilizarse -- los únicos usos del término son para declarar explícitamente 'uses_ground_truth: False' y rechazar cualquier intento de usarlo")
def test_scount_07_ground_truth_never_used():
    fixture = ThematicFixture()
    result = fixture.agent().execute(fixture.agent_input(min_sections=1, max_sections=1, enforce_section_count=False))
    manifest_path = result.output_artifacts["manifest"].path
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["safety_policy"]["uses_ground_truth"] is False

    import inspect

    from src.tools.thematic_analysis import coverage_validation, prompting

    for module in (coverage_validation, prompting):
        assert "ground_truth" not in inspect.getsource(module).lower()


if __name__ == "__main__":
    for fn in (
        test_scount_01_enforce_true_too_many_sections_blocks,
        test_scount_02_enforce_false_too_many_sections_approved,
        test_scount_03_enforce_false_too_few_sections_does_not_block,
        test_scount_04_section_metrics_present_in_audit,
        test_scount_05_word_length_limits_untouched,
        test_scount_06_other_real_failures_still_block,
        test_scount_07_ground_truth_never_used,
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
