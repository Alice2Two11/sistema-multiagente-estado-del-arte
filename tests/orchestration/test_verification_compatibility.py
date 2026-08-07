"""Pruebas de `validate_agent07_orchestrator_compatibility` y del retriever portado.

No llama a OpenAI ni a Chroma real (usa un doble de colección determinista
para el retriever). Corre como script plano.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.adapters.verification_orchestrator_runtime import (
    validate_agent07_orchestrator_compatibility,
)
from src.adapters.verification_incremental_retriever import Agent07ChromaRetriever

RESULTS = []


def scenario(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
            except Exception:  # noqa: BLE001
                RESULTS.append((name, False, traceback.format_exc()))
            else:
                RESULTS.append((name, True, ""))

        return wrapper

    return decorator


def _base_kwargs(project_dir: Path, experiment_id: str):
    experiment_dir = project_dir / experiment_id
    return dict(
        project_dir=project_dir,
        experiment_dir=experiment_dir,
        experiment_paths={
            "project_root": str(project_dir.resolve()),
            "experiment_root": str(experiment_dir.resolve()),
        },
        active_experiment_config={
            "active_experiment_id": experiment_id,
            "openai_model": "gpt-4o-mini",
            "verification_policy": {"p": 1},
            "verification_prompt_version": "v3",
            "verification_budgets": {"max_llm_attempts": 2},
        },
        agent07_config={
            "verification_model": "gpt-4o-mini",
            "correction_model": "gpt-4o-mini",
            "verification_policy": {"p": 1},
            "verification_prompt_version": "v3",
            "verification_budgets": {"max_llm_attempts": 2},
        },
    )


@scenario("1. configuración compatible: no lanza")
def test_compatible_configuration():
    project_dir = Path("/tmp/proj_a")
    kwargs = _base_kwargs(project_dir, "exp1")
    validate_agent07_orchestrator_compatibility(**kwargs)  # no debe lanzar


@scenario("2. experimento incompatible: agent07_config divergió del config activo")
def test_incompatible_experiment_config_drift():
    project_dir = Path("/tmp/proj_a")
    kwargs = _base_kwargs(project_dir, "exp1")
    kwargs["agent07_config"] = dict(kwargs["agent07_config"])
    kwargs["agent07_config"]["verification_model"] = "gpt-3.5-turbo"  # divergió
    try:
        validate_agent07_orchestrator_compatibility(**kwargs)
    except ValueError as exc:
        assert "AGENT07_GLOBAL_CONFIG_MISMATCH:verification_model" in str(exc)
    else:
        raise AssertionError("debía rechazar la divergencia de verification_model")


@scenario("3. rutas distintas pero semánticamente válidas: pasan si son internamente consistentes")
def test_different_but_valid_paths():
    for project_dir in (Path("/tmp/proj_a"), Path("/tmp/otro/lugar/muy/distinto")):
        kwargs = _base_kwargs(project_dir, "exp1")
        validate_agent07_orchestrator_compatibility(**kwargs)  # ninguna debe lanzar


@scenario("4. artefactos de otro experimento: experiment_dir no coincide con active_experiment_id")
def test_artifacts_from_another_experiment():
    project_dir = Path("/tmp/proj_a")
    kwargs = _base_kwargs(project_dir, "exp1")
    # experiment_dir apunta a "exp2" pero active_experiment_id sigue siendo "exp1"
    kwargs["experiment_dir"] = project_dir / "exp2"
    kwargs["experiment_paths"] = {
        "project_root": str(project_dir.resolve()),
        "experiment_root": str((project_dir / "exp2").resolve()),
    }
    try:
        validate_agent07_orchestrator_compatibility(**kwargs)
    except ValueError as exc:
        assert "AGENT07_EXPERIMENT_ID_MISMATCH" in str(exc)
    else:
        raise AssertionError("debía rechazar experiment_dir/active_experiment_id inconsistentes")


@scenario("4b. retriever real rechaza un manifest de Chroma de otro experimento")
def test_retriever_rejects_foreign_chroma_manifest():
    from src.adapters.verification_incremental_retriever import (
        build_agent07_chroma_retriever,
    )
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        chroma_manifest = tmp / "chroma_index_manifest.json"
        chroma_manifest.write_text(
            json.dumps({"experiment_id": "OTRO_EXPERIMENTO"}), encoding="utf-8"
        )
        chunks_manifest = tmp / "chunks_clean_for_rag.jsonl"
        chunks_manifest.write_text("{}\n", encoding="utf-8")
        try:
            build_agent07_chroma_retriever(
                chroma_dir=tmp,
                chroma_collection_name="reference_papers_chunks",
                embedding_model_name="all-MiniLM-L6-v2",
                chroma_manifest_path=chroma_manifest,
                chunks_manifest_path=chunks_manifest,
                committed_experiment_id="exp1",
                rag_policy={},
            )
        except ValueError as exc:
            assert "pertenece a otro experimento" in str(exc)
        else:
            raise AssertionError("debía rechazar el manifest de otro experimento")


@scenario("5. Agent07ChromaRetriever: comportamiento ante ausencia de evidencia")
def test_retriever_no_evidence_behavior():
    class EmptyCollection:
        def query(self, *, query_texts, n_results):
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    retriever = Agent07ChromaRetriever(
        collection=EmptyCollection(),
        experiment_id="exp1",
        collection_name="c",
        embedding_model="m",
        chroma_manifest_fingerprint="f1",
        chunks_manifest_fingerprint="f2",
    )
    result = retriever.retrieve_more(
        {
            "claim_id": "c1",
            "claim_context": {"claim_text": "algo"},
            "allowed_source_filenames": ["paper.pdf"],
        }
    )
    assert result["selected_candidates"] == ()
    assert result["stop_reason"] == "NO_NEW_EVIDENCE"
    assert result["structural_coverage_improved"] is False


@scenario("6. Agent07ChromaRetriever: filtra por fuentes autorizadas y respeta top_k")
def test_retriever_filters_and_caps():
    class FakeCollection:
        def query(self, *, query_texts, n_results):
            return {
                "documents": [["texto A", "texto B", "texto C"]],
                "metadatas": [[
                    {"source_filename": "autorizado.pdf", "chunk_id": "c0"},
                    {"source_filename": "no_autorizado.pdf", "chunk_id": "c1"},
                    {"source_filename": "autorizado.pdf", "chunk_id": "c2"},
                ]],
                "distances": [[0.1, 0.2, 0.3]],
            }

    retriever = Agent07ChromaRetriever(
        collection=FakeCollection(),
        experiment_id="exp1",
        collection_name="c",
        embedding_model="m",
        chroma_manifest_fingerprint="f1",
        chunks_manifest_fingerprint="f2",
        top_k=1,
    )
    result = retriever.retrieve_more(
        {
            "claim_id": "c1",
            "claim_context": {"claim_text": "algo"},
            "allowed_source_filenames": ["autorizado.pdf"],
        }
    )
    assert len(result["selected_candidates"]) == 1, result
    assert result["selected_candidates"][0]["source_filename"] == "autorizado.pdf"
    assert result["selected_candidates"][0]["chunk_id"] == "c0"


if __name__ == "__main__":
    for fn in (
        test_compatible_configuration,
        test_incompatible_experiment_config_drift,
        test_different_but_valid_paths,
        test_artifacts_from_another_experiment,
        test_retriever_rejects_foreign_chroma_manifest,
        test_retriever_no_evidence_behavior,
        test_retriever_filters_and_caps,
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
