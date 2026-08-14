"""Fase 1 de aislamiento del contrato canónico ``sentences[]`` (V2):
demuestra -- no asume -- que la bifurcación mínima introducida en
``draft_writing_agent.py`` (constantes ``LEGACY_DRAFT_REPRESENTATION_
CONTRACT``/``CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT`` + un
único ``if`` antes del bucle de intentos) no altera absolutamente nada
del comportamiento legacy: mismo ``AgentResult``, mismo ``draft_text``,
mismos ``claims[]``, mismos validation reports, mismo fingerprint
histórico, misma transición del orquestador, cero invocaciones al
código V2, y los mismos artefactos observables por los consumidores
reales de 07/08.

El código legacy en sí (rondas 1-5, todo el cuerpo del bucle de
generación) NO se tocó: no se movió, no se extrajo a una función
nueva, no se reindentó. Solo se agregó un ``if`` antes de su primera
línea. Estos tests comparan la ejecución real contra un snapshot
congelado de ``draft_writing_agent.py`` tal como estaba ANTES de esta
fase, cargado dinámicamente bajo un nombre de módulo distinto -- no
contra una descripción de lo que "debería" pasar."""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "v16"))

from test_agent06_v16 import Env  # noqa: E402

from src.adapters.draft_writing_runtime import DraftWritingRuntime, _draft_signature  # noqa: E402
from src.agents.draft_writing_agent import (  # noqa: E402
    CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT,
    LEGACY_DRAFT_REPRESENTATION_CONTRACT,
    DraftWritingAgent,
)
from src.state.fingerprints import fingerprint_mapping  # noqa: E402

PRECHANGE_SNAPSHOT = (
    Path(__file__).resolve().parent / "fixtures" / "draft_writing_agent_v17round5_snapshot.py"
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


def _load_precompare_agent_class():
    """Carga el snapshot PRE-CAMBIO de draft_writing_agent.py bajo un
    nombre de módulo distinto (nunca reemplaza el módulo real en
    sys.modules) -- permite ejecutar el código EXACTO de antes de esta
    fase, lado a lado con el código actual, en el mismo proceso."""

    spec = importlib.util.spec_from_file_location(
        "draft_writing_agent_precompare", PRECHANGE_SNAPSHOT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DraftWritingAgent


_MOCK_RESPONSE = json.dumps({
    "section_id": "S1", "section_title": "Methods",
    "draft_text": "The method reports an accuracy of 95 percent measured in the study [a.pdf | c1].",
    "claims": [{
        "claim": "The method reports an accuracy of 95 percent measured in the study",
        "supporting_citations": ["[a.pdf | c1]"],
    }],
})


def _invoke(prompt):
    return _MOCK_RESPONSE


def _run_precompare(env):
    PrecompareAgent = _load_precompare_agent_class()
    agent = PrecompareAgent(DraftWritingRuntime(_invoke, env.collection))
    return agent.execute(env.ai)


def _run_current(env, contract=None):
    ai = env.ai
    if contract is not None:
        from dataclasses import replace
        ai = replace(ai, policy={**ai.policy, "draft_representation_contract": contract})
    agent = DraftWritingAgent(DraftWritingRuntime(_invoke, env.collection))
    return agent.execute(ai)


@scenario("LEGACY01. AgentResult equivalente: código pre-cambio vs código actual con flag AUSENTE")
def test_legacy01_agent_result_equivalent():
    env_pre = Env(attempt=1)
    result_pre = _run_precompare(env_pre)
    env_now = Env(attempt=1)
    result_now = _run_current(env_now, contract=None)

    assert result_pre.execution_status == result_now.execution_status
    assert result_pre.quality_status == result_now.quality_status
    assert result_pre.decision.code == result_now.decision.code
    assert result_pre.decision.rationale == result_now.decision.rationale
    assert set(result_pre.output_artifacts.keys()) == set(result_now.output_artifacts.keys())


@scenario("LEGACY02. draft_text idéntico byte a byte")
def test_legacy02_identical_draft_text():
    env_pre = Env(attempt=1)
    _run_precompare(env_pre)
    env_now = Env(attempt=1)
    _run_current(env_now, contract=None)

    draft_pre = json.loads((env_pre.out / "state_of_art_draft.json").read_text())
    draft_now = json.loads((env_now.out / "state_of_art_draft.json").read_text())
    section_pre = next(s for s in draft_pre["sections"] if s["section_id"] == "S1")
    section_now = next(s for s in draft_now["sections"] if s["section_id"] == "S1")
    assert section_pre["draft_text"] == section_now["draft_text"]


@scenario("LEGACY03. claims[] idénticos (estructura y contenido, no solo semántica)")
def test_legacy03_identical_claims():
    env_pre = Env(attempt=1)
    _run_precompare(env_pre)
    env_now = Env(attempt=1)
    _run_current(env_now, contract=None)

    draft_pre = json.loads((env_pre.out / "state_of_art_draft.json").read_text())
    draft_now = json.loads((env_now.out / "state_of_art_draft.json").read_text())
    section_pre = next(s for s in draft_pre["sections"] if s["section_id"] == "S1")
    section_now = next(s for s in draft_now["sections"] if s["section_id"] == "S1")
    assert section_pre["claims"] == section_now["claims"]


@scenario("LEGACY04. Validation reports idénticos (incluida instrumentación retry_audit de ronda 5)")
def test_legacy04_identical_validation_reports():
    env_pre = Env(attempt=1)
    _run_precompare(env_pre)
    env_now = Env(attempt=1)
    _run_current(env_now, contract=None)

    v_pre = json.loads((env_pre.out / "raw_section_outputs" / "agent_attempt_01" / "S1_attempt_1_validation.json").read_text())
    v_now = json.loads((env_now.out / "raw_section_outputs" / "agent_attempt_01" / "S1_attempt_1_validation.json").read_text())
    # raw_output_path incluye la ruta absoluta del directorio temporal
    # de CADA ejecución (distinto por diseño, no por regresión) --
    # se excluye de la comparación exacta.
    v_pre.pop("raw_output_path", None)
    v_now.pop("raw_output_path", None)
    assert v_pre == v_now


@scenario("LEGACY05. Fingerprint histórico idéntico: código anterior == flag ausente == 'legacy' explícito")
def test_legacy05_fingerprint_identical_across_three_cases():
    base_policy = {
        "stage_version": "S", "prompt_version": "P", "rag_version": "R", "validation_version": "V",
        "normalization_version": "N",
    }
    cfg_common = {
        "experiment_id": "e", "experiment_dir": "d", "model": "m",
        "embedding_model_name": "em", "chroma_collection_name": "c",
    }

    # Caso 1: firma "pre-cambio" -- reconstruida sin la clave nueva en
    # absoluto (equivalente exacto de lo que _draft_signature producía
    # antes de esta fase, ya que esa versión nunca conocía la clave).
    fp_precompare = fingerprint_mapping({
        "stage": "06_agente_redactor", "stage_version": base_policy["stage_version"],
        "experiment_id": cfg_common["experiment_id"], "experiment_dir": cfg_common["experiment_dir"],
        "openai_model": cfg_common["model"], "embedding_model_name": cfg_common["embedding_model_name"],
        "chroma_collection_name": cfg_common["chroma_collection_name"],
        "topic_profile": {}, "experiment_profile": {}, "generation_profile": {}, "rag_policy": {},
        "draft_generation_policy": base_policy,
        "paths": {}, "hashes": {},
        "prompt_version": base_policy["prompt_version"], "rag_version": base_policy["rag_version"],
        "validation_version": base_policy["validation_version"],
    })

    # Caso 2: código actual, flag AUSENTE.
    fp_absent = fingerprint_mapping(_draft_signature({**cfg_common, "policy": dict(base_policy)}, {}))

    # Caso 3: código actual, "legacy" EXPLÍCITO en la policy real.
    fp_explicit_legacy = fingerprint_mapping(_draft_signature(
        {**cfg_common, "policy": {**base_policy, "draft_representation_contract": LEGACY_DRAFT_REPRESENTATION_CONTRACT}}, {},
    ))

    assert fp_precompare == fp_absent == fp_explicit_legacy

    # Control negativo: V2 SÍ debe producir un fingerprint distinto.
    fp_v2 = fingerprint_mapping(_draft_signature(
        {**cfg_common, "policy": {**base_policy, "draft_representation_contract": CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT}}, {},
    ))
    assert fp_v2 != fp_absent


@scenario("LEGACY06. Misma transición del orquestador (action/reason_code/target_stage)")
def test_legacy06_identical_orchestrator_transition():
    env_pre = Env(attempt=1)
    result_pre = _run_precompare(env_pre)
    env_now = Env(attempt=1)
    result_now = _run_current(env_now, contract=None)

    t_pre = result_pre.requested_transition
    t_now = result_now.requested_transition
    assert t_pre.action == t_now.action
    assert t_pre.reason_code == t_now.reason_code
    assert t_pre.target_stage == t_now.target_stage


@scenario("LEGACY07. Cero invocaciones al camino V2 durante una corrida legacy (flag ausente y 'legacy' explícito)")
def test_legacy07_zero_v2_invocations():
    import src.tools.draft_writing.canonical_sentences as canonical_module

    calls = {"n": 0}
    original = canonical_module.generate_section_canonical_v2

    def traced(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    canonical_module.generate_section_canonical_v2 = traced
    try:
        env_absent = Env(attempt=1)
        _run_current(env_absent, contract=None)
        env_explicit = Env(attempt=1)
        _run_current(env_explicit, contract=LEGACY_DRAFT_REPRESENTATION_CONTRACT)
    finally:
        canonical_module.generate_section_canonical_v2 = original

    assert calls["n"] == 0


@scenario("LEGACY07b. El módulo canonical_sentences NO se importa durante una corrida legacy (verificado quitándolo de sys.modules primero)")
def test_legacy07b_module_not_imported_during_legacy_run():
    module_name = "src.tools.draft_writing.canonical_sentences"
    was_present = module_name in sys.modules
    if was_present:
        removed = sys.modules.pop(module_name)
    else:
        removed = None
    try:
        env = Env(attempt=1)
        _run_current(env, contract=None)
        # Si la corrida legacy hubiera importado el módulo V2 en
        # cualquier punto, reaparecería en sys.modules -- no debe.
        assert module_name not in sys.modules
    finally:
        if was_present:
            sys.modules[module_name] = removed


@scenario("LEGACY09. Flag desconocido: falla explícitamente (fail-closed), no ejecuta legacy ni V2, y tampoco produce un fingerprint")
def test_legacy09_unknown_contract_fails_closed():
    from dataclasses import replace

    env = Env(attempt=1)
    ai = replace(env.ai, policy={**env.ai.policy, "draft_representation_contract": "some_future_contract_v3"})
    agent = DraftWritingAgent(DraftWritingRuntime(_invoke, env.collection))
    result = agent.execute(ai)

    assert result.execution_status.value == "FAILED"
    assert result.error is not None
    assert "UNKNOWN_DRAFT_REPRESENTATION_CONTRACT:some_future_contract_v3" in str(result.error)
    assert not (env.out / "state_of_art_draft.json").exists()
    assert not list((env.out / "raw_section_outputs").rglob("S1_attempt_1.txt"))

    base_policy = {
        "stage_version": "S", "prompt_version": "P", "rag_version": "R", "validation_version": "V",
        "normalization_version": "N", "draft_representation_contract": "some_future_contract_v3",
    }
    cfg_common = {
        "experiment_id": "e", "experiment_dir": "d", "model": "m",
        "embedding_model_name": "em", "chroma_collection_name": "c",
    }
    try:
        _draft_signature({**cfg_common, "policy": base_policy}, {})
    except ValueError as exc:
        assert "UNKNOWN_DRAFT_REPRESENTATION_CONTRACT:some_future_contract_v3" in str(exc)
    else:
        raise AssertionError("un contrato desconocido debe fallar cerrado también en _draft_signature")


@scenario("LEGACY09b. Primera sección SOURCE-FREE (organizativa, sin evidencia) + contrato desconocido -> falla ANTES de procesarla, cero invocaciones V2, ningún draft final")
def test_legacy09b_unknown_contract_fails_before_source_free_section():
    from dataclasses import replace

    env = Env(attempt=1)
    # Outline reescrito: la sección source-free (conclusion, sin
    # papers_to_use) queda PRIMERA -- exactamente el caso que antes
    # podía "colarse" (generated.append(...); continue) antes de que
    # la validación del contrato se moviera fuera del bucle.
    outline_path = env.inp / "outline.json"
    outline = json.loads(outline_path.read_text())
    outline["sections"] = list(reversed(outline["sections"]))
    assert outline["sections"][0]["section_type"] == "conclusion"
    outline_path.write_text(json.dumps(outline), encoding="utf-8")

    ai = replace(env.ai, policy={**env.ai.policy, "draft_representation_contract": "some_future_contract_v3"})

    import src.tools.draft_writing.canonical_sentences as canonical_module
    calls = {"n": 0}
    original = canonical_module.generate_section_canonical_v2

    def traced(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    canonical_module.generate_section_canonical_v2 = traced
    try:
        agent = DraftWritingAgent(DraftWritingRuntime(_invoke, env.collection))
        result = agent.execute(ai)
    finally:
        canonical_module.generate_section_canonical_v2 = original

    # Falla explícitamente, ANTES de procesar la primera sección
    # (source-free o no) -- nunca llega a generar nada para ninguna.
    assert result.execution_status.value == "FAILED"
    assert "UNKNOWN_DRAFT_REPRESENTATION_CONTRACT:some_future_contract_v3" in str(result.error)
    # Cero invocaciones a V2 -- ni siquiera para la sección source-free,
    # que bajo el código anterior a esta corrección podía procesarse
    # ANTES de alcanzar la validación del contrato.
    assert calls["n"] == 0
    # Ningún draft final se generó -- ni siquiera el manifest parcial
    # que produciría una sección source-free procesada exitosamente.
    assert not (env.out / "state_of_art_draft.json").exists()
    assert not (env.out / "draft_generation_manifest.json").exists()


@scenario("LEGACY08. El consumidor REAL de 07 (build_agent07_input_from_committed_agent06) procesa idénticamente un draft producido por el código pre-cambio y por el código actual en modo legacy -- pipeline completo, sin mocks del consumidor")
def test_legacy08_same_artifacts_for_real_07_consumers():
    import tempfile
    from src.adapters.draft_writing_runtime import (
        build_draft_agent_input,
        load_draft_configuration,
        prepare_draft_execution,
        execute_prepared_draft,
        commit_executed_draft,
    )
    from src.adapters.agent06_verification_handoff import (
        build_agent07_input_from_committed_agent06,
    )

    single_section = [{
        "section_id": "S1", "section_title": "Sec1", "section_type": "linea_tematica",
        "purpose": "p", "key_arguments": ["a"], "evidence_needs": ["e"],
        "papers_to_use": [{"source_filename": "paper_a.pdf", "title": "A"}],
    }]
    response = json.dumps({
        "section_id": "S1", "section_title": "Sec1",
        "draft_text": "The observed accuracy value reaches ninety one percent across the reported comparative baseline measurements [paper_a.pdf | a_chroma].",
        "claims": [{
            "claim": "The observed accuracy value reaches ninety one percent across the reported comparative baseline measurements",
            "supporting_citations": ["[paper_a.pdf | a_chroma]"],
        }],
    })

    def run_full_pipeline(agent_class):
        tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(tmp.name)
        experiment, store, chunk_rows = write_synthetic_project(
            tmp_path, sections=single_section, quantitative="none"
        )
        cfg = load_draft_configuration(
            tmp_path, attempt_number=1, chroma_client_factory=chroma_client_factory,
        )
        collection = SyntheticCollection(chunk_rows)
        runtime = DraftWritingRuntime(lambda p: response, collection)
        agent = agent_class(runtime)
        agent_input = build_draft_agent_input(cfg)

        prepared = prepare_draft_execution(store=store, agent_input=agent_input)
        executed = execute_prepared_draft(store=store, agent=agent, prepared=prepared)
        assert executed.result.quality_status.value.startswith("APPROVED"), executed.result.decision
        commit_executed_draft(store=store, executed=executed)

        handoff = build_agent07_input_from_committed_agent06(
            store=store, stage_name="06_agente_redactor",
            agent07_config={}, policy_versions={}, schema_versions={},
            experiment_paths={"experiment_dir": str(experiment)},
            outline_paper_mapping_path=experiment / "05_outputs" / "04_outline" / "outline_paper_mapping.csv",
        )
        tmp.cleanup()
        return handoff

    from agent06_v17_test_support import SyntheticCollection, chroma_client_factory, write_synthetic_project

    PrecompareAgent = _load_precompare_agent_class()
    handoff_pre = run_full_pipeline(PrecompareAgent)
    handoff_now = run_full_pipeline(DraftWritingAgent)

    # Campos que 07 REALMENTE consume del handoff -- excluidos
    # deliberadamente los que son identificadores de sesión/ejecución
    # (run_id, agent06_decision_id) y por tanto legítimamente distintos
    # entre dos ejecuciones separadas, aunque el CONTENIDO sea idéntico.
    for key in (
        "claim_verification_contexts",
        "expected_claim_ids",
        "claim_inventory_fingerprint",
        "source_draft_fingerprint",
    ):
        assert handoff_pre[key] == handoff_now[key], key


@scenario("LEGACY08b. Consumidor real de 08 -- no ejecutado en esta fase; documentado por qué, sin afirmar cobertura que no existe")
def test_legacy08b_agent08_consumer_documented_scope():
    # NUNCA se incluye en el for-fn de ejecución de abajo -- no cuenta
    # como test ejecutado, ni como PASS ni como FAIL. Existe solo como
    # documentación explícita de alcance, tal como se pidió.
    #
    # Agent08 consume artefactos de 07 COMMITTED (evaluation_upstream.py,
    # build_agent08_input_from_committed_agent07) -- requiere, como
    # precondición real, un ciclo COMPLETO de verificación de 07 ya
    # comprometido (con claim_evidence_traceability_rows reales,
    # producto de una verificación LLM real o mockeada con la misma
    # profundidad que 07 exige). Montar eso aquí exigiría replicar el
    # aparato completo de Agent07 (verify_claim, resolution,
    # traceability bundle) -- contexto que excede el alcance de esta
    # fase, centrada en el aislamiento de 06. No se ejecuta ningún
    # consumidor real de 08 en este test -- 08 queda cubierto
    # ÚNICAMENTE por la igualdad contractual ya demostrada en LEGACY02/
    # LEGACY03 (draft_text/claims[] idénticos, que es lo único que 08
    # podría llegar a ver indirectamente vía 07). Este test no afirma
    # haber ejecutado código real de 08 -- solo dejar constancia
    # explícita del alcance, tal como se pidió.
    pass


if __name__ == "__main__":
    for fn in (
        test_legacy01_agent_result_equivalent,
        test_legacy02_identical_draft_text,
        test_legacy03_identical_claims,
        test_legacy04_identical_validation_reports,
        test_legacy05_fingerprint_identical_across_three_cases,
        test_legacy06_identical_orchestrator_transition,
        test_legacy07_zero_v2_invocations,
        test_legacy07b_module_not_imported_during_legacy_run,
        test_legacy09_unknown_contract_fails_closed,
        test_legacy09b_unknown_contract_fails_before_source_free_section,
        test_legacy08_same_artifacts_for_real_07_consumers,
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
    # LEGACY08b NUNCA se cuenta como test ejecutado -- es documentación
    # explícita de alcance (ver su docstring): Agent08 no fue montado
    # en esta fase, y no se afirma cobertura que no existe.
    print("Agent08 consumer: NOT EXECUTED / documented scope (ver LEGACY08b)")
    raise SystemExit(1 if failed else 0)
