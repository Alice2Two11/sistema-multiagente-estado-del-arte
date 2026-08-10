"""``claim_identity_migration_signal`` (parche 17) tenía consumidor
(``verification_notebook.py``) pero ningún productor -- ``draft_
writing_agent.py`` nunca lo emitía. Corrección: el productor real es
``build_agent07_input_from_committed_agent06`` (``agent06_verification_
handoff.py``) -- la única función que construye ``committed_agent06_
output`` a partir del draft comprometido y del ``CycleState`` real.

La señal es ``True`` EXCLUSIVAMENTE cuando, a la vez:
1. el ciclo escrito tiene declarado el contrato ``LEGACY``;
2. TODOS los claims publicados en el draft comprometido tienen
   ``claim_version`` válido (``>=1``, entero) -- confirma que se
   produjeron vía el mecanismo real de identidad, no un ``claim_uid``
   puesto a mano;
3. TODOS tienen además ``claim_uid`` no vacío.

Nunca configurada a mano por experimento; nunca inferida por
similitud; máximo una vez (una vez declarado ``STABLE_UID_V1``, nunca
vuelve a ser ``True``)."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "orchestration"))

import test_verification_stagespec_integration as T  # noqa: E402

from src.adapters.agent06_verification_handoff import build_agent07_input_from_committed_agent06  # noqa: E402
from src.contracts.agent_input import ArtifactReference  # noqa: E402
from src.contracts.agent_result import (  # noqa: E402
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.state.fingerprints import sha256_file  # noqa: E402
from src.state.pipeline_state import (  # noqa: E402
    ArtifactState,
    CycleState,
    DecisionLogEntry,
    PipelineIdentity,
    PipelineState,
)
from src.state.state_store import StateStore  # noqa: E402

RESULTS = []


def scenario(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
            except Exception:
                import traceback

                RESULTS.append((name, False, traceback.format_exc()))
            else:
                RESULTS.append((name, True, ""))

        return wrapper

    return decorator


def _seed_project_with_claim_identity(tmp_path: Path, *, add_identity_to_claims: bool, drop_one_uid: bool = False):
    """Variante de ``T._seed_project`` que inyecta ``claim_uid``/
    ``claim_version`` reales en TODOS los claims del draft comprometido
    (fixture real, no inventado) antes de calcular su hash -- para
    poder ejercitar ``build_agent07_input_from_committed_agent06`` bajo
    los distintos escenarios de contrato de identidad."""

    root = tmp_path / "proj"
    root.mkdir()
    experiment_id = "exp_stagespec"
    (root / "active_experiment.json").write_text(
        json.dumps({"active_experiment_id": experiment_id, "run_id": "run_stagespec"}), encoding="utf-8"
    )
    experiment_dir = root / experiment_id
    outputs = experiment_dir / "05_outputs"
    outputs.mkdir(parents=True)
    state_path = outputs / "00_orchestrator_planner" / "pipeline_state.json"
    state_path.parent.mkdir(parents=True)

    names = (
        "state_of_art_draft.json", "state_of_art_draft.md", "draft_sections.csv",
        "draft_rag_evidence.csv", "draft_claim_evidence.csv", "numeric_hallucination_check.csv",
        "draft_validation_report.json", "draft_generation_manifest.json",
    )
    artifacts_dir = tmp_path / "agent06_artifacts"
    artifacts_dir.mkdir()
    refs = {}
    for name in names:
        source_bytes = (T.FIXTURE_DIR / name).read_bytes()
        if name == "state_of_art_draft.json":
            draft = json.loads(source_bytes)
            claim_index = 0
            for section in draft.get("sections", []):
                for claim in section.get("claims", []):
                    claim_index += 1
                    if add_identity_to_claims and not (drop_one_uid and claim_index == 1):
                        claim["claim_uid"] = f"uid-real-{claim_index:03d}"
                        claim["claim_version"] = 1
                        claim["parent_claim_uids"] = []
                        claim["claim_text_fingerprint"] = "f" * 64
                        claim["created_round"] = 1
                        claim["updated_round"] = 1
            source_bytes = json.dumps(draft, ensure_ascii=False).encode("utf-8")
        target = artifacts_dir / name
        target.write_bytes(source_bytes)
        refs[name] = ArtifactReference(str(target), sha256_file(target))

    now = "2026-01-01T00:00:00+00:00"
    result = AgentResult(
        execution_status=ExecutionStatus.COMPLETED, quality_status=QualityStatus.APPROVED,
        decision=DecisionInfo("OK", "ok"), quality_metrics={}, warnings=(),
        requested_transition=RequestedTransition(TransitionAction.ADVANCE, T.VERIFY, "OK", False),
        output_artifacts=refs, tool_usage=ToolUsage(), attempt_number=1, started_at=now, completed_at=now,
    )
    log = DecisionLogEntry("d06", now, T.DRAFT, T.DRAFT, 1, {}, {"code": "OK"}, (), None, result.to_dict())
    state = PipelineState(
        identity=PipelineIdentity(experiment_id, "run_stagespec", now, now, "v1"),
        stages={T.DRAFT: T.StageState(execution_status=ExecutionStatus.COMPLETED)},
        artifacts={name: ArtifactState(ref, now) for name, ref in refs.items()},
        decision_log=(log,),
    )
    store = StateStore(state_path)
    store.initialize(state)

    outline_dir = outputs / "04_outline"
    outline_dir.mkdir(parents=True)
    mapping_path = outline_dir / "outline_paper_mapping.csv"
    mapping_path.write_bytes((T.FIXTURE_DIR / "outline_paper_mapping.csv").read_bytes())

    return root, store, mapping_path


def _declare_cycle(store, *, contract_version):
    state = store.load()
    cycle = CycleState(rounds_used=1, max_rounds=3, status="ACTIVE", claim_identity_contract_version=contract_version)
    store.save(replace(state, cycles={**state.cycles, "writer_verifier": cycle}))


def _build_handoff(store, project_dir, mapping_path):
    return build_agent07_input_from_committed_agent06(
        store=store, stage_name=T.DRAFT, agent07_config={}, policy_versions={"verification": "v1"},
        schema_versions={"runtime": "v5", "provisional_bundle": "v4", "multi_proposal_resolution": "v1"},
        experiment_paths={"root": str(project_dir)}, outline_paper_mapping_path=mapping_path,
    )


@scenario("X01. LEGACY declarado + salida de 06 completa con claim_uid/claim_version en TODOS -> signal=True")
def test_legacy_full_identity_signal_true():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = _seed_project_with_claim_identity(Path(tmp), add_identity_to_claims=True)
        _declare_cycle(store, contract_version="LEGACY")
        handoff = _build_handoff(store, project_dir, mapping_path)
        assert handoff["claim_identity_migration_signal"] is True


@scenario("X02. LEGACY declarado + claim_uid incompleto (falta en uno) -> signal=False (no publica migración)")
def test_legacy_incomplete_identity_signal_false():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = _seed_project_with_claim_identity(
            Path(tmp), add_identity_to_claims=True, drop_one_uid=True,
        )
        _declare_cycle(store, contract_version="LEGACY")
        handoff = _build_handoff(store, project_dir, mapping_path)
        assert handoff["claim_identity_migration_signal"] is False


@scenario("X03. STABLE_UID_V1 declarado (revisión posterior a la migración) -> signal=False, aunque los claims tengan identidad completa")
def test_stable_revision_signal_false():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = _seed_project_with_claim_identity(Path(tmp), add_identity_to_claims=True)
        _declare_cycle(store, contract_version="STABLE_UID_V1")
        handoff = _build_handoff(store, project_dir, mapping_path)
        assert handoff["claim_identity_migration_signal"] is False


@scenario("X04. Sin ciclo todavía (draft inicial nuevo, ya STABLE) -> signal=False, porque no es una migración -- es la primera frontera")
def test_no_cycle_yet_new_stable_signal_false():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = _seed_project_with_claim_identity(Path(tmp), add_identity_to_claims=True)
        # Sin declarar ningún cycle -- el ciclo no existe todavía.
        handoff = _build_handoff(store, project_dir, mapping_path)
        assert handoff["claim_identity_migration_signal"] is False


@scenario("X05. Integración real: 06 en revisión LEGACY -> committed_agent06_output con signal=True -> 07 migra el ciclo a STABLE_UID_V1")
def test_integration_legacy_revision_to_migration():
    from src.tools.verification.writer_revision_cycle import classify_verification_transition

    with tempfile.TemporaryDirectory() as tmp:
        project_dir, store, mapping_path = _seed_project_with_claim_identity(Path(tmp), add_identity_to_claims=True)
        _declare_cycle(store, contract_version="LEGACY")
        handoff = _build_handoff(store, project_dir, mapping_path)
        assert handoff["claim_identity_migration_signal"] is True

        # 07 usa esa señal real (no una construida a mano) para decidir
        # si migra -- mismo camino que execute_prepared_agent07.
        claims = [
            {"claim_id": ctx["claim_id"], "claim_uid": ctx["claim_uid"], "final_correction_eligibility": "NO_CORRECTION_NEEDED", "evidence_used": ()}
            for ctx in handoff["claim_verification_contexts"]
        ]
        cycle = store.load().cycles["writer_verifier"]
        decision = classify_verification_transition(
            claims=claims, technical_status="COMPLETED", rounds_used=cycle.rounds_used, max_rounds=cycle.max_rounds,
            declared_contract_version=cycle.claim_identity_contract_version,
            migration_signal=handoff["claim_identity_migration_signal"],
        )
        assert decision["claim_identity_contract_migrated"] is True
        assert decision["claim_identity_contract_version"] == "STABLE_UID_V1"


if __name__ == "__main__":
    for fn in (
        test_legacy_full_identity_signal_true,
        test_legacy_incomplete_identity_signal_false,
        test_stable_revision_signal_false,
        test_no_cycle_yet_new_stable_signal_false,
        test_integration_legacy_revision_to_migration,
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
