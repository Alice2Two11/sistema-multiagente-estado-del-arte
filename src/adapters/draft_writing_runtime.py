from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from src.agents.draft_writing_agent import (
    CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT,
    LEGACY_DRAFT_REPRESENTATION_CONTRACT,
    DraftWritingAgent,
)
from src.config.draft_writing_policy_config import (
    LEGACY_RETRIEVAL_STRATEGY,
    PLANNED_HYBRID_RETRIEVAL_STRATEGY,
    get_draft_writing_policy,
)
from src.contracts.agent_input import (
    AgentContext,
    AgentInput,
    ArtifactReference,
    ExecutionMode,
    PreviousAttemptSummary,
)
from src.contracts.agent_result import AgentResult, QualityStatus
from src.runtime.draft_writing_protocol import build_draft_fingerprints
from src.state.fingerprints import fingerprint_mapping, sha256_file
from src.utils.json_parsing import parse_json_safely


LEGACY_RUNTIME_VERSIONS = {
    "stage_version": "06_AGENTIC_V16_BEHAVIOR_PRESERVING",
    "rag_version": "legacy_chroma_then_csv_restricted_v1",
    # v2 (sufijo "_hard_word_range_configured_min_gate_soft_failure_
    # length_repair"): el gate de longitud total del borrador cambió de
    # contrato -- antes, global_length_valid podía aprobarse usando
    # effective_min_total_words (rebajado silenciosamente por número de
    # secciones source-free), permitiendo que un borrador muy por
    # debajo de min_total_words configurado (ej. 1081 con
    # configured_min=1300) se aprobara. Desde esta versión:
    # configured_min_total_words/configured_max_total_words (los
    # valores reales del generation_profile) son el único gate;
    # incumplir el rango produce reason codes explícitos
    # (TOTAL_WORD_COUNT_BELOW_MINIMUM/ABOVE_MAXIMUM/INSUFFICIENT_
    # SUPPORTED_CONTENT_FOR_MIN_LENGTH) en vez de INVALID_DRAFT
    # genérico, nunca una excepción técnica, y se intenta una
    # reparación dirigida (src/tools/draft_writing/length_repair.py,
    # exclusivamente dentro del contrato Evidence Handles V2) antes de
    # agotar los intentos. Este es un cambio de CONTRATO real -- por
    # eso participa en el fingerprint (ver _draft_signature) e invalida
    # cualquier draft/manifest de 06 producido bajo una versión
    # anterior, sin necesitar --force-rerun.
    "validation_version": "legacy_notebook06_validation_v2_hard_word_range_configured_min_gate_soft_failure_length_repair",
    # Contrato de normalización oración<->claim (normalize_generated_
    # section, src/tools/draft_writing/normalization.py). Se incrementa
    # explícitamente cada vez que cambia CÓMO se decide preservar/
    # heredar/descartar una oración -- nunca se infiere de un hash de
    # archivo. Antes de esta versión: una oración sustantiva sin
    # correspondencia EXACTA con un claim se BORRABA por completo,
    # pudiendo vaciar draft_text entero (EMPTY_DRAFT_TEXT). Desde esta
    # versión: se preserva sin cita inventada, dejando que la
    # validación reporte el motivo real. Este cambio de CONTRATO
    # (no solo de implementación) invalida cualquier draft/manifest de
    # 06 producido bajo una versión anterior -- por eso participa en el
    # fingerprint (ver _draft_signature) y en la validación de resume
    # (ver _manifest_versions_match), sin necesitar --force-rerun.
    #
    # v2 (sufijo "_immediate_numeric_salvage_v2"): el algoritmo de
    # ACEPTACIÓN de una sección también cambió -- _salvage_numeric_
    # only_section ahora se intenta INMEDIATAMENTE tras CADA intento
    # interno cuya validación falla EXCLUSIVAMENTE por UNSUPPORTED_
    # NUMERIC_VALUE:*, no solo una vez al final sobre el último intento
    # (logs[-1]). Un intento intermedio numeric-only-salvageable ya no
    # se pierde silenciosamente si un intento POSTERIOR vuelve a fallar
    # con errores mixtos -- una sección que antes terminaba
    # SECTION_VALIDATION_FAILED ahora puede aceptarse. Mismo criterio
    # fail-closed heredado sin cambios (_unsupported_numeric_values):
    # cualquier error de cita/claim/estructura sigue desactivando el
    # salvage por completo, y nunca se reemplaza/inventa un valor
    # numérico -- solo se descartan oraciones/claims que lo contienen.
    #
    # v3 (sufijo "_discourse_connector_feedback_v3"): dos cambios
    # observables más, ninguno relaja el matcher exacto ni acepta una
    # sección desalineada:
    #   1. build_section_prompt (regla 11) ahora incluye un ejemplo
    #      positivo/negativo explícito del patrón de fallo real más
    #      común (conector discursivo inicial omitido en claims[].claim)
    #      -- el prompt en sí cambió, así que el MISMO input upstream
    #      puede producir una respuesta distinta del LLM.
    #   2. detect_claims_missing_leading_discourse_connector (nuevo,
    #      normalization.py) agrega feedback ESPECÍFICO y accionable a
    #      previous_errors cuando detecta -- de forma estrictamente
    #      determinista, sin fuzzy/semantic matching -- que un claim
    #      desalineado difiere de su oración únicamente por el conector
    #      discursivo inicial. Nunca cambia validation_ok, nunca hereda
    #      ni inventa una cita, nunca repara el claim generado: la
    #      sección sigue rechazándose exactamente igual que antes ante
    #      el mismo input -- pero el feedback más específico en el
    #      RETRY puede cambiar qué intento termina siendo aceptado en
    #      una corrida real, por eso participa en el fingerprint.
    "normalization_version": "sentence_claim_exact_match_preserve_unmatched_v1_immediate_numeric_salvage_v2_discourse_connector_feedback_v3",
}
HYBRID_RUNTIME_VERSIONS = {
    "stage_version": "06_AGENTIC_V17_HYBRID_QUANTITATIVE_SOURCE_AWARE",
    "rag_version": "hybrid_chroma_csv_rrf_balanced_v1",
    "quantitative_selection_version": "confirmed_literal_greedy_coverage_v1",
    "budget_version": "source_aware_exact_total_v1",
    # v2 (sufijo "_hard_word_range_configured_min_gate_soft_failure_
    # length_repair"): el gate de longitud total del borrador cambió de
    # contrato -- antes, global_length_valid podía aprobarse usando
    # effective_min_total_words (rebajado silenciosamente por número de
    # secciones source-free), permitiendo que un borrador muy por
    # debajo de min_total_words configurado (ej. 1081 con
    # configured_min=1300) se aprobara. Desde esta versión:
    # configured_min_total_words/configured_max_total_words (los
    # valores reales del generation_profile) son el único gate;
    # incumplir el rango produce reason codes explícitos
    # (TOTAL_WORD_COUNT_BELOW_MINIMUM/ABOVE_MAXIMUM/INSUFFICIENT_
    # SUPPORTED_CONTENT_FOR_MIN_LENGTH) en vez de INVALID_DRAFT
    # genérico, nunca una excepción técnica, y se intenta una
    # reparación dirigida (src/tools/draft_writing/length_repair.py,
    # exclusivamente dentro del contrato Evidence Handles V2) antes de
    # agotar los intentos. Este es un cambio de CONTRATO real -- por
    # eso participa en el fingerprint (ver _draft_signature) e invalida
    # cualquier draft/manifest de 06 producido bajo una versión
    # anterior, sin necesitar --force-rerun.
    "validation_version": "legacy_notebook06_validation_v2_hard_word_range_configured_min_gate_soft_failure_length_repair",
    "normalization_version": "sentence_claim_exact_match_preserve_unmatched_v1_immediate_numeric_salvage_v2_discourse_connector_feedback_v3",
}
REQUIRED_DRAFT_ARTIFACTS = (
    "state_of_art_draft.json",
    "state_of_art_draft.md",
    "draft_sections.csv",
    "draft_rag_evidence.csv",
    "draft_quality_check.csv",
    "draft_length_check.csv",
    "draft_claim_evidence.csv",
    "numeric_hallucination_check.csv",
    "draft_validation_report.json",
    "quantitative_comparative_table_used.csv",
    "dataset_technique_summary_used.csv",
    "draft_generation_manifest.json",
)


@dataclass
class DraftWritingRuntime:
    invoke_fn: object
    collection: object

    def invoke(self, prompt):
        return self.invoke_fn(prompt)

    def parse(self, raw):
        # Extracción robusta del texto real de la respuesta -- reutiliza
        # parse_json_safely (src/tools/evaluation/llm_judge.py, ya
        # probado y usado en producción por 03B/08) en vez de duplicar
        # lógica de extracción de JSON: maneja dict/list ya parseados,
        # fences ```json ... ``` y ``` ... ```, y JSON puro, escaneando
        # con json.JSONDecoder().raw_decode desde cualquier {/[ -- más
        # robusto que buscar el primer '{' y el último '}' a mano (evita
        # capturar basura tras el objeto JSON real).
        content = getattr(raw, "content", raw)
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            # Bloques de contenido estructurado del proveedor (ej. Anthropic/
            # OpenAI "content blocks": [{"type": "text", "text": "..."}]) --
            # se concatena el texto de cada bloque, nunca se serializa la
            # lista completa como texto crudo (eso rompería el parseo).
            parts: list[str] = []
            for block in content:
                if isinstance(block, Mapping):
                    block_text = block.get("text")
                    if isinstance(block_text, str):
                        parts.append(block_text)
                elif isinstance(block, str):
                    parts.append(block)
            content = "\n".join(parts)
        try:
            parsed = parse_json_safely(content)
        except Exception as exc:
            # Fail-closed: nunca se convierte una respuesta inválida en
            # un dict vacío ni en draft_text="" -- se propaga un error
            # explícito, igual que antes.
            raise ValueError("INVALID_LLM_OUTPUT") from exc
        if not isinstance(parsed, dict):
            raise ValueError("INVALID_LLM_OUTPUT")
        return parsed


def build_openai_draft_runtime(
    model,
    temperature,
    collection,
    *,
    project_dir=None,
    llm_factory=None,
    human_message_factory=None,
):
    from src.io.credentials import load_runtime_credential

    load_runtime_credential("OPENAI_API_KEY", project_dir=project_dir)
    if llm_factory is None:
        from langchain_openai import ChatOpenAI

        llm_factory = ChatOpenAI
    if human_message_factory is None:
        from langchain_core.messages import HumanMessage

        human_message_factory = HumanMessage
    llm = llm_factory(model=model, temperature=float(temperature))
    return DraftWritingRuntime(
        lambda prompt: llm.invoke([human_message_factory(content=prompt)]).content,
        collection,
    )


def _runtime_versions(strategy: str) -> dict[str, str]:
    if strategy == PLANNED_HYBRID_RETRIEVAL_STRATEGY:
        return dict(HYBRID_RUNTIME_VERSIONS)
    if strategy == LEGACY_RETRIEVAL_STRATEGY:
        return dict(LEGACY_RUNTIME_VERSIONS)
    raise ValueError(f"UNSUPPORTED_DRAFT_RETRIEVAL_STRATEGY:{strategy}")


def build_runtime_draft_policy(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a coherent runtime policy while keeping the general default legacy."""
    requested = dict(overrides or {})
    policy = get_draft_writing_policy(requested)
    strategy = str(policy["retrieval_strategy"])
    policy.update(_runtime_versions(strategy))
    if strategy == LEGACY_RETRIEVAL_STRATEGY:
        policy.pop("quantitative_selection_version", None)
        policy.pop("budget_version", None)
    return policy


def resolve_pipeline_state_path(project_dir, experiment_id):
    root = Path(project_dir).resolve()
    exp = root / experiment_id
    canonical = exp / "05_outputs" / "00_orchestrator_planner" / "pipeline_state.json"
    if canonical.is_file():
        return canonical
    candidates = list(exp.rglob("pipeline_state.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"pipeline_state.json no encontrado en {exp}")
    raise RuntimeError(f"pipeline_state.json ambiguo: {candidates}")


def _collection_name(item):
    name = getattr(item, "name", None)
    return str(name if name is not None else item)


def _open_chroma_client(path, client_factory=None):
    if client_factory is None:
        import chromadb

        client_factory = chromadb.PersistentClient
    try:
        return client_factory(path=str(path))
    except TypeError:
        return client_factory(str(path))


def resolve_chroma_dir(
    experiment_dir,
    expected_collection,
    explicit_path=None,
    *,
    client_factory=None,
):
    exp = Path(experiment_dir).resolve()
    ordered = []
    if explicit_path:
        ordered.append(Path(explicit_path).expanduser())
    ordered.append(exp / "04_chroma_index")
    ordered.extend(sorted({p.parent for p in exp.rglob("chroma.sqlite3")}, key=str))
    candidates = []
    seen = set()
    for item in ordered:
        path = item if item.is_absolute() else exp / item
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(Path(key))
    valid = []
    observed = {}
    for path in candidates:
        if not path.is_dir() or not (path / "chroma.sqlite3").is_file():
            continue
        try:
            client = _open_chroma_client(path, client_factory)
            names = sorted({_collection_name(item) for item in client.list_collections()})
        except Exception as exc:
            observed[str(path)] = [f"CLIENT_ERROR:{type(exc).__name__}"]
            continue
        observed[str(path)] = names
        if expected_collection in names:
            valid.append(path.resolve())
    unique = []
    seen_valid = set()
    for path in valid:
        key = str(path)
        if key not in seen_valid:
            seen_valid.add(key)
            unique.append(path)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError(f"CHROMA_DIR_AMBIGUOUS:{[str(item) for item in unique]}")
    raise FileNotFoundError(
        f"CHROMA_COLLECTION_NOT_FOUND:{expected_collection}; observed={observed}"
    )


def load_draft_configuration(
    project_dir,
    attempt_number=1,
    *,
    chroma_client_factory=None,
    policy_overrides: Mapping[str, Any] | None = None,
):
    root = Path(project_dir).resolve()
    active = json.loads((root / "active_experiment.json").read_text(encoding="utf-8"))
    experiment_id = active["active_experiment_id"]
    experiment_dir = root / experiment_id
    outputs = experiment_dir / "05_outputs"
    outline = outputs / "04_outline"
    thematic = outputs / "03_thematic_analysis"
    draft = outputs / "05_draft"
    rag = active.get("rag_policy", {})
    active_policy = dict(active.get("draft_generation_policy", {}))
    if policy_overrides:
        active_policy.update(dict(policy_overrides))
    policy = build_runtime_draft_policy(active_policy)
    generation = active.get("generation_profile", {})
    policy.update(
        {
            "experiment_profile": active.get("experiment_profile", {}),
            "topic_profile": active.get("topic_profile", {}),
            "generation_profile": generation,
            "rag_policy": rag,
            "output_language": generation.get("output_language", "español académico"),
            "writing_mode": generation.get("writing_mode", ""),
            "focus_mode": generation.get("focus_mode", ""),
            "citation_style": generation.get("citation_style", ""),
            "target_total_words": int(generation.get("target_total_words", 1000)),
            "min_total_words": int(generation.get("min_total_words", 650)),
            "max_total_words": int(generation.get("max_total_words", 1400)),
        }
    )
    collection_name = active.get("chroma_collection_name", "reference_papers_chunks")
    chroma_dir = resolve_chroma_dir(
        experiment_dir,
        collection_name,
        active.get("chroma_dir"),
        client_factory=chroma_client_factory,
    )
    paths = {
        "outline_json": outline / "state_of_art_outline.json",
        "outline_mapping": outline / "outline_paper_mapping.csv",
        "outline_validation": outline / "outline_validation_report.json",
        "outline_manifest": outline / "outline_generation_manifest.json",
        "kb_final": thematic / "kb_final_for_thematic_analysis.csv",
        "thematic_manifest": thematic / "thematic_analysis_manifest.json",
        "thematic_validation": thematic / "thematic_validation_report.json",
        "chunks_clean": Path(
            active.get(
                "chunks_clean_path",
                experiment_dir / "03_chunks" / "chunks_clean_for_rag.csv",
            )
        ),
        "chroma_manifest": Path(
            active.get(
                "chroma_manifest_path",
                outputs / "01_rag" / "chroma_index_manifest.json",
            )
        ),
        "quantitative_table": outputs
        / "02_scientific_knowledge_base"
        / "quantitative_comparative_table.csv",
        "dataset_summary": outputs
        / "02_scientific_knowledge_base"
        / "dataset_technique_summary.csv",
        "quantitative_manifest": outputs
        / "02_scientific_knowledge_base"
        / "quantitative_extraction_manifest.json",
    }
    return {
        "project_dir": root,
        "experiment_id": experiment_id,
        "run_id": active.get("run_id", experiment_id),
        "attempt_number": int(attempt_number),
        "model": active.get("openai_model", "gpt-4o-mini"),
        "embedding_model_name": active.get(
            "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "chroma_collection_name": collection_name,
        "chroma_dir": chroma_dir,
        "policy": policy,
        "output_dir": draft,
        "state_path": resolve_pipeline_state_path(root, experiment_id),
        "paths": paths,
        "experiment_dir": experiment_dir,
    }


def _previous_draft_attempt(cfg):
    if cfg["attempt_number"] != 2:
        return None
    payload = json.loads(Path(cfg["state_path"]).read_text(encoding="utf-8"))
    stage = payload.get("stages", {}).get("06_agente_redactor", {})
    if stage.get("requested_transition", {}).get("action") != "RETRY":
        raise RuntimeError("El intento 2 requiere una transición RETRY persistida.")
    return PreviousAttemptSummary(
        quality_status=stage.get("quality_status", "NEEDS_REVISION"),
        failure_reason_codes=tuple(stage.get("failure_reason_codes", [])),
        blocking_warnings=tuple(
            str(item.get("code", ""))
            for item in stage.get("warnings", [])
            if item.get("blocking") and item.get("code")
        ),
        previous_artifacts={},
    )


def _dependency_references(cfg) -> dict[str, ArtifactReference]:
    required = (
        "outline_json",
        "outline_mapping",
        "outline_validation",
        "outline_manifest",
        "kb_final",
        "thematic_manifest",
        "thematic_validation",
        "chunks_clean",
    )
    dependencies = {}
    for name in required:
        path = Path(cfg["paths"][name])
        if not path.is_file():
            raise FileNotFoundError(f"{name} no existe: {path}")
        dependencies[name] = ArtifactReference(str(path), sha256_file(path))
    chroma_manifest = Path(cfg["paths"]["chroma_manifest"])
    if chroma_manifest.is_file():
        dependencies["chroma_manifest"] = ArtifactReference(
            str(chroma_manifest), sha256_file(chroma_manifest)
        )
    optional = ("quantitative_table", "dataset_summary", "quantitative_manifest")
    present = [name for name in optional if Path(cfg["paths"][name]).is_file()]
    if present and len(present) != len(optional):
        raise FileNotFoundError("INVALID_QUANTITATIVE_CONTEXT")
    for name in present:
        path = Path(cfg["paths"][name])
        dependencies[name] = ArtifactReference(str(path), sha256_file(path))
    return dependencies


def _draft_signature(cfg, dependencies) -> dict[str, Any]:
    policy = {key: value for key, value in cfg["policy"].items() if key != "current_fingerprint"}
    contract = policy.get(
        "draft_representation_contract",
        LEGACY_DRAFT_REPRESENTATION_CONTRACT,
    )
    if contract not in {
        LEGACY_DRAFT_REPRESENTATION_CONTRACT,
        CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT,
    }:
        # Misma semántica fail-closed que en draft_writing_agent.py: un
        # valor desconocido nunca puede colapsar silenciosamente al
        # fingerprint legacy -- se rechaza aquí también, antes de
        # construir ninguna firma.
        raise ValueError(f"UNKNOWN_DRAFT_REPRESENTATION_CONTRACT:{contract}")
    if contract != CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT:
        # Legacy (ausente o "legacy" explícito): la clave NUNCA
        # participa en el fingerprint histórico -- se remueve
        # explícitamente de la copia usada para la firma, sin importar
        # si ya estaba presente en cfg["policy"]. Esto es lo único que
        # garantiza fingerprint(legacy) == fingerprint(pre-cambio),
        # byte a byte, incluso si "legacy" llega a aparecer explícito
        # en una configuración real en el futuro.
        policy = {key: value for key, value in policy.items() if key != "draft_representation_contract"}
    signature = {
        "stage": "06_agente_redactor",
        "stage_version": policy["stage_version"],
        "experiment_id": cfg["experiment_id"],
        "experiment_dir": str(cfg["experiment_dir"]),
        "openai_model": cfg["model"],
        "embedding_model_name": cfg["embedding_model_name"],
        "chroma_collection_name": cfg["chroma_collection_name"],
        "topic_profile": policy.get("topic_profile", {}),
        "experiment_profile": policy.get("experiment_profile", {}),
        "generation_profile": policy.get("generation_profile", {}),
        "rag_policy": policy.get("rag_policy", {}),
        "draft_generation_policy": policy,
        "paths": {key: value.path for key, value in dependencies.items()},
        "hashes": {key: value.hash for key, value in dependencies.items()},
        "prompt_version": policy["prompt_version"],
        "rag_version": policy["rag_version"],
        "validation_version": policy["validation_version"],
    }
    if contract == CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT:
        # ÚNICAMENTE V2 incorpora esta clave al fingerprint -- nunca
        # legacy, ni por ausencia ni por valor explícito (ver arriba).
        signature["draft_representation_contract"] = CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT
    return signature


def build_draft_agent_input(cfg):
    dependencies = _dependency_references(cfg)
    signature = _draft_signature(cfg, dependencies)
    policy = dict(cfg["policy"])
    policy["current_fingerprint"] = fingerprint_mapping(signature)
    cfg["policy"] = policy
    return AgentInput(
        experiment_id=cfg["experiment_id"],
        run_id=cfg["run_id"],
        stage_name="06_agente_redactor",
        attempt_number=cfg["attempt_number"],
        mode=ExecutionMode.FULL_RUN,
        agent_context=AgentContext(
            allowed_tools=(
                "llm",
                "chroma",
                "csv_retrieval",
                "atomic_write",
                "draft_validation",
            ),
            output_directory=str(cfg["output_dir"]),
            runtime_resources={
                "model": cfg["model"],
                "chroma_collection_name": cfg["chroma_collection_name"],
                "embedding_model_name": cfg["embedding_model_name"],
                "chroma_dir": str(cfg["chroma_dir"]),
            },
        ),
        dependencies=dependencies,
        policy=policy,
        previous_attempt=_previous_draft_attempt(cfg),
    )


def build_chroma_collection(cfg):
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(cfg["chroma_dir"]))
    embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg["embedding_model_name"]
    )
    return client.get_collection(
        name=cfg["chroma_collection_name"], embedding_function=embedding
    )


def build_real_draft_execution(
    project_dir,
    attempt_number=1,
    *,
    collection_factory=None,
    runtime_factory=None,
    chroma_client_factory=None,
    policy_overrides: Mapping[str, Any] | None = None,
):
    cfg = load_draft_configuration(
        project_dir,
        attempt_number,
        chroma_client_factory=chroma_client_factory,
        policy_overrides=policy_overrides,
    )
    collection = (collection_factory or build_chroma_collection)(cfg)
    runtime_builder = runtime_factory or build_openai_draft_runtime
    runtime = runtime_builder(
        cfg["model"],
        cfg["policy"]["temperature"],
        collection,
        project_dir=cfg["project_dir"],
    )
    return DraftWritingAgent(runtime), build_draft_agent_input(cfg), cfg


@dataclass(frozen=True)
class PreparedDraftExecution:
    decision_id: str
    agent_input: AgentInput


@dataclass(frozen=True)
class ExecutedDraftExecution:
    decision_id: str
    agent_input: AgentInput
    result: AgentResult
    persisted_result_path: str


def prepare_draft_execution(*, store, agent_input: AgentInput) -> PreparedDraftExecution:
    prepared = store.prepare_execution(
        target_stage=agent_input.stage_name,
        intended_action="EXECUTE_DRAFT_WRITING",
        attempt_number=agent_input.attempt_number,
    )
    return PreparedDraftExecution(prepared.decision_id, agent_input)


def execute_prepared_draft(
    *,
    store,
    agent,
    prepared: PreparedDraftExecution,
) -> ExecutedDraftExecution:
    result = agent.execute(prepared.agent_input)
    path = store.persist_agent_result(prepared.decision_id, result)
    return ExecutedDraftExecution(
        prepared.decision_id,
        prepared.agent_input,
        result,
        str(path),
    )


def _approved_result(result: AgentResult) -> bool:
    return result.quality_status in {
        QualityStatus.APPROVED,
        QualityStatus.APPROVED_WITH_WARNINGS,
        QualityStatus.APPROVED_AFTER_MANUAL_REVIEW,
    }


def _validate_complete_artifacts(result: AgentResult) -> None:
    missing = [name for name in REQUIRED_DRAFT_ARTIFACTS if name not in result.output_artifacts]
    if missing:
        raise RuntimeError(f"DRAFT_COMMIT_INCOMPLETE_ARTIFACTS:{','.join(missing)}")


def commit_executed_draft(*, store, executed: ExecutedDraftExecution, observations=None):
    if not _approved_result(executed.result):
        raise RuntimeError("DRAFT_COMMIT_REQUIRES_APPROVED_RESULT")
    _validate_complete_artifacts(executed.result)
    return store.commit_execution(
        decision_id=executed.decision_id,
        result=executed.result,
        stage_name=executed.agent_input.stage_name,
        fingerprints=build_draft_fingerprints(executed.agent_input),
        observations=dict(observations or {}),
    )


def _manifest_versions_match(manifest: Mapping[str, Any], expected_versions: Mapping[str, str]) -> bool:
    versions = manifest.get("versions", {})
    if not isinstance(versions, Mapping):
        return False
    if (
        versions.get("stage") != expected_versions["stage_version"]
        or versions.get("rag") != expected_versions["rag_version"]
        or versions.get("validation") != expected_versions["validation_version"]
        # Un manifest producido ANTES de este contrato no tiene la clave
        # "normalization" en absoluto (versions.get(...) da None) --
        # nunca coincide con el string real esperado, invalidando
        # correctamente cualquier resume/reuse sobre un draft generado
        # con la normalización anterior (que podía vaciar draft_text).
        or versions.get("normalization") != expected_versions["normalization_version"]
    ):
        return False
    if "quantitative_selection_version" in expected_versions:
        if "retrieval_strategy" not in manifest:
            return True
        return (
            versions.get("quantitative_selection")
            == expected_versions["quantitative_selection_version"]
            and versions.get("budget") == expected_versions["budget_version"]
        )
    return "quantitative_selection" not in versions and "budget" not in versions


def _is_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "")))


def _validate_resume_candidate(
    *,
    result: AgentResult,
    agent_input: AgentInput,
    artifact_references: Mapping[str, ArtifactReference] | None = None,
) -> None:
    if not _approved_result(result):
        raise RuntimeError("DRAFT_RESUME_REQUIRES_APPROVED_RESULT")

    missing = [
        name for name in REQUIRED_DRAFT_ARTIFACTS
        if name not in result.output_artifacts
    ]
    if missing:
        raise RuntimeError(
            f"DRAFT_COMMIT_INCOMPLETE_ARTIFACTS:{','.join(missing)}"
        )

    committed_refs = dict(artifact_references or {})
    for name in REQUIRED_DRAFT_ARTIFACTS:
        reference = result.output_artifacts[name]
        path = Path(reference.path)
        if not path.is_file():
            if name == "draft_generation_manifest.json":
                raise RuntimeError("DRAFT_RESUME_MANIFEST_NOT_FOUND")
            raise RuntimeError(f"DRAFT_COMMIT_INCOMPLETE_ARTIFACTS:{name}")
        if committed_refs:
            committed = committed_refs.get(name)
            if committed is None:
                raise RuntimeError(f"DRAFT_COMMIT_INCOMPLETE_ARTIFACTS:{name}")
            if committed.path != reference.path or committed.hash != reference.hash:
                raise RuntimeError(f"DRAFT_RESUME_ARTIFACT_REFERENCE_MISMATCH:{name}")
        if _is_sha256(reference.hash) and sha256_file(path) != reference.hash:
            raise RuntimeError(f"DRAFT_RESUME_ARTIFACT_HASH_MISMATCH:{name}")

    manifest_path = Path(
        result.output_artifacts["draft_generation_manifest.json"].path
    )
    if not manifest_path.is_file():
        raise RuntimeError("DRAFT_RESUME_MANIFEST_NOT_FOUND")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("DRAFT_RESUME_MANIFEST_INVALID") from exc
    if not isinstance(manifest, Mapping):
        raise RuntimeError("DRAFT_RESUME_MANIFEST_INVALID")

    strategy = str(agent_input.policy["retrieval_strategy"])
    expected_versions = _runtime_versions(strategy)
    expected_fingerprint = str(
        agent_input.policy.get("current_fingerprint", "")
    )

    if "retrieval_strategy" in manifest:
        if manifest.get("retrieval_strategy") != strategy:
            raise RuntimeError("DRAFT_RESUME_VERSION_MISMATCH")
        if not _manifest_versions_match(manifest, expected_versions):
            raise RuntimeError("DRAFT_RESUME_VERSION_MISMATCH")
        if (
            not expected_fingerprint
            or manifest.get("fingerprint") != expected_fingerprint
        ):
            raise RuntimeError("DRAFT_RESUME_FINGERPRINT_MISMATCH")
        return

    if not expected_fingerprint or manifest.get("fingerprint") != expected_fingerprint:
        raise RuntimeError("DRAFT_RESUME_FINGERPRINT_MISMATCH")
    if not _manifest_versions_match(manifest, expected_versions):
        raise RuntimeError("DRAFT_RESUME_VERSION_MISMATCH")


def _committed_result_from_state(state, stage_name: str) -> AgentResult:
    for entry in reversed(state.decision_log):
        if entry.stage != stage_name:
            continue
        try:
            return AgentResult.from_dict(entry.result)
        except Exception as exc:
            raise RuntimeError("DRAFT_RESUME_RESULT_INVALID") from exc
    raise RuntimeError("DRAFT_RESUME_RESULT_NOT_FOUND")


def _committed_artifact_references(state) -> dict[str, ArtifactReference]:
    return {
        name: artifact_state.reference
        for name, artifact_state in state.artifacts.items()
    }


def resume_draft_execution(*, store, agent_input: AgentInput, observations=None):
    state = store.load()
    pending = state.pending_execution

    if pending is not None:
        result = store.find_persisted_agent_result(pending.decision_id)
        if result is None:
            return store.resolve_resume(
                stage_name=agent_input.stage_name,
                fingerprints=build_draft_fingerprints(agent_input),
                observations=dict(observations or {}),
            )
        try:
            _validate_resume_candidate(
                result=result,
                agent_input=agent_input,
            )
        except Exception:
            store.cancel_pending_execution()
            raise
        return store.resolve_resume(
            stage_name=agent_input.stage_name,
            fingerprints=build_draft_fingerprints(agent_input),
            observations=dict(observations or {}),
        )

    result = _committed_result_from_state(state, agent_input.stage_name)
    _validate_resume_candidate(
        result=result,
        agent_input=agent_input,
        artifact_references=_committed_artifact_references(state),
    )
    from src.state.state_store import ResumeResolution

    return ResumeResolution(
        action="NO_PENDING",
        state=state,
        committed_result=result,
    )
