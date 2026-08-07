from copy import deepcopy
import json
from pathlib import Path
import pytest

from test_phase72_runtime_notebook_closure import deps
from test_phase73_transactional_integration import store_at, tx_input
from src.adapters.verification_runtime import build_agent07_runtime_dependencies
from src.adapters.verification_notebook import (
    AGENT07_ARTIFACT_NAMES,
    commit_executed_agent07,
    execute_prepared_agent07,
    prepare_agent07_execution,
    resume_agent07_execution,
)


class DummyLLM:
    def invoke(self, messages):
        return "{}"


def productive_config():
    return {
        "verification_policy": {"mode": "strict"},
        "correction_policy": {"mode": "strict"},
        "reverification_policy": {"mode": "strict"},
        "verification_prompt_version": "v1",
        "correction_prompt_version": "v1",
        "reverification_prompt_version": "v1",
        "verification_budgets": {"max_llm_attempts": 1},
        "correction_budgets": {"max_llm_attempts": 1},
        "reverification_budgets": {"max_llm_attempts": 1},
    }


def test_productive_factory_uses_real_components_and_no_fixture_imports():
    cfg = productive_config()
    built = build_agent07_runtime_dependencies(
        config=cfg,
        experiment_paths={"root": "/tmp/exp"},
        verification_llm=DummyLLM(),
        correction_llm=DummyLLM(),
        reverification_llm=DummyLLM(),
    )
    assert built.verification_agent_factory.__module__ == "src.agents.verification_agent"
    assert built.proposal_runner.__module__ == "src.tools.verification.corrections"
    assert built.bundle_builder.__module__ == "src.tools.verification.validation"
    assert "test" not in built.correction_context_factory.__module__


def test_productive_factory_missing_required_parameter_blocks():
    cfg = productive_config(); del cfg["correction_prompt_version"]
    with pytest.raises(ValueError, match="correction_prompt_version"):
        build_agent07_runtime_dependencies(
            config=cfg, experiment_paths={"root":"/tmp/exp"},
            verification_llm=DummyLLM(), correction_llm=DummyLLM(), reverification_llm=DummyLLM(),
        )


def test_crash_after_release_before_state_commit_is_resumed_without_reexecution(tmp_path, monkeypatch):
    store = store_at(tmp_path)
    value = tx_input(tmp_path)
    prepared = prepare_agent07_execution(store=store, runtime_input=value)
    executed = execute_prepared_agent07(store=store, prepared=prepared, dependencies=deps("COMPLETED"))
    original_commit = store.commit_execution
    calls = {"count": 0}
    def fail_once(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("crash-after-publish")
        return original_commit(**kwargs)
    monkeypatch.setattr(store, "commit_execution", fail_once)
    with pytest.raises(RuntimeError, match="crash-after-publish"):
        commit_executed_agent07(store=store, executed=executed)
    official = Path(value.experiment_paths["agent07_output_dir"])
    assert all((official/name).is_file() for name in AGENT07_ARTIFACT_NAMES)
    resumed = resume_agent07_execution(store=store, runtime_input=value)
    assert resumed.action == "COMMITTED"
    state = store.load()
    assert state.pending_execution is None
    assert len([x for x in state.decision_log if x.stage == "07_agente_verificador"]) == 1


def test_productive_notebook_does_not_initialize_pipeline_state():
    nb = json.loads(Path("notebooks/07_agente_verificador_trazabilidad_LIMPIO.ipynb").read_text())
    source = "\n".join("".join(c.get("source", ())) for c in nb["cells"])
    productive = source.split("if FIXTURE_MODE:", 1)[-1]
    assert "build_agent07_runtime_dependencies" in source
    assert "resolve_committed_agent06_output" in source
    assert "if not state_path.exists():\n    store.initialize" not in source
