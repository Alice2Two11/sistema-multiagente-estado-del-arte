from pathlib import Path
import json

# ============================================================
# CONFIGURACIÓN GENERAL DEL PROYECTO
# Fuente única para rutas, experimento activo, modelos y políticas.
# ============================================================

PROJECT_DIR = Path("/content/proyecto_estado_arte")
SRC_DIR = PROJECT_DIR / "src"
ACTIVE_EXPERIMENT_PATH = (
    PROJECT_DIR
    / "active_experiment.json"
)


def _load_active_experiment():
    if not ACTIVE_EXPERIMENT_PATH.exists():
        raise FileNotFoundError(
            "No existe active_experiment.json. "
            "Ejecuta primero 00_setup_config."
        )

    try:
        with ACTIVE_EXPERIMENT_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)
    except Exception as error:
        raise RuntimeError(
            "No se pudo leer active_experiment.json."
        ) from error

    if not isinstance(data, dict):
        raise TypeError(
            "active_experiment.json debe contener un objeto JSON."
        )

    required_keys = {
        "active_experiment_id",
        "experiment_dir",
        "generation_profile",
        "topic_profile",
        "openai_model",
        "embedding_model",
        "chroma_collection_name",
        "rag_policy",
        "extraction_policy",
        "quantitative_extraction_policy",
        "thematic_analysis_policy",
        "ingestion_policy",
        "outline_generation_policy",
        "draft_generation_policy",
        "verification_policy",
        "post_correction_recheck_policy",
        "evaluation_policy",
    }

    missing = sorted(
        required_keys - set(data)
    )

    if missing:
        raise ValueError(
            "active_experiment.json está incompleto. "
            f"Faltan: {missing}"
        )

    return data


ACTIVE_EXPERIMENT = (
    _load_active_experiment()
)

EXPERIMENT_ID = str(
    ACTIVE_EXPERIMENT[
        "active_experiment_id"
    ]
).strip()

if not EXPERIMENT_ID:
    raise ValueError(
        "active_experiment_id no puede estar vacío."
    )

EXPERIMENT_DIR = Path(
    ACTIVE_EXPERIMENT[
        "experiment_dir"
    ]
)

expected_experiment_dir = (
    PROJECT_DIR / EXPERIMENT_ID
)

if (
    EXPERIMENT_DIR.resolve()
    != expected_experiment_dir.resolve()
):
    raise ValueError(
        "experiment_dir no coincide con active_experiment_id."
    )

GROUND_TRUTH_DIR = (
    EXPERIMENT_DIR
    / "00_ground_truth"
)
INPUT_PDFS_DIR = (
    EXPERIMENT_DIR
    / "01_input_references_pdfs"
)
EXTRACTED_TEXTS_DIR = (
    EXPERIMENT_DIR
    / "02_extracted_texts"
)
CHUNKS_DIR = (
    EXPERIMENT_DIR
    / "03_chunks"
)
CHROMA_DIR = (
    EXPERIMENT_DIR
    / "04_chroma_index"
)
OUTPUTS_DIR = (
    EXPERIMENT_DIR
    / "05_outputs"
)

SCIENTIFIC_EXTRACTION_DIR = (
    OUTPUTS_DIR
    / "01_scientific_extraction"
)
KNOWLEDGE_BASE_DIR = (
    OUTPUTS_DIR
    / "02_scientific_knowledge_base"
)
THEMATIC_ANALYSIS_DIR = (
    OUTPUTS_DIR
    / "03_thematic_analysis"
)
OUTLINE_DIR = (
    OUTPUTS_DIR
    / "04_outline"
)
DRAFT_DIR = (
    OUTPUTS_DIR
    / "05_draft"
)
VERIFICATION_TRACEABILITY_DIR = (
    OUTPUTS_DIR
    / "06_verification_traceability"
)
EVALUATION_DIR = (
    OUTPUTS_DIR
    / "07_evaluation"
)
ORCHESTRATOR_DIR = (
    OUTPUTS_DIR
    / "00_orchestrator_planner"
)

GENERATION_PROFILE = (
    ACTIVE_EXPERIMENT[
        "generation_profile"
    ]
)
TOPIC_PROFILE = (
    ACTIVE_EXPERIMENT[
        "topic_profile"
    ]
)
EMBEDDING_MODEL_NAME = str(
    ACTIVE_EXPERIMENT[
        "embedding_model"
    ]
).strip()
OPENAI_MODEL = str(
    ACTIVE_EXPERIMENT[
        "openai_model"
    ]
).strip()
CHROMA_COLLECTION_NAME = str(
    ACTIVE_EXPERIMENT[
        "chroma_collection_name"
    ]
).strip()

RAG_POLICY = (
    ACTIVE_EXPERIMENT[
        "rag_policy"
    ]
)
EXTRACTION_POLICY = (
    ACTIVE_EXPERIMENT[
        "extraction_policy"
    ]
)
QUANTITATIVE_EXTRACTION_POLICY = (
    ACTIVE_EXPERIMENT[
        "quantitative_extraction_policy"
    ]
)
THEMATIC_ANALYSIS_POLICY = (
    ACTIVE_EXPERIMENT[
        "thematic_analysis_policy"
    ]
)
INGESTION_POLICY = (
    ACTIVE_EXPERIMENT[
        "ingestion_policy"
    ]
)
OUTLINE_GENERATION_POLICY = (
    ACTIVE_EXPERIMENT[
        "outline_generation_policy"
    ]
)
DRAFT_GENERATION_POLICY = (
    ACTIVE_EXPERIMENT[
        "draft_generation_policy"
    ]
)
VERIFICATION_POLICY = (
    ACTIVE_EXPERIMENT[
        "verification_policy"
    ]
)
POST_CORRECTION_RECHECK_POLICY = (
    ACTIVE_EXPERIMENT[
        "post_correction_recheck_policy"
    ]
)
EVALUATION_POLICY = (
    ACTIVE_EXPERIMENT[
        "evaluation_policy"
    ]
)

for name, value in {
    "GENERATION_PROFILE": GENERATION_PROFILE,
    "TOPIC_PROFILE": TOPIC_PROFILE,
    "RAG_POLICY": RAG_POLICY,
    "EXTRACTION_POLICY": EXTRACTION_POLICY,
    "QUANTITATIVE_EXTRACTION_POLICY": (
        QUANTITATIVE_EXTRACTION_POLICY
    ),
    "THEMATIC_ANALYSIS_POLICY": (
        THEMATIC_ANALYSIS_POLICY
    ),
    "INGESTION_POLICY": INGESTION_POLICY,
    "OUTLINE_GENERATION_POLICY": (
        OUTLINE_GENERATION_POLICY
    ),
    "DRAFT_GENERATION_POLICY": (
        DRAFT_GENERATION_POLICY
    ),
    "VERIFICATION_POLICY": (
        VERIFICATION_POLICY
    ),
    "POST_CORRECTION_RECHECK_POLICY": (
        POST_CORRECTION_RECHECK_POLICY
    ),
    "EVALUATION_POLICY": (
        EVALUATION_POLICY
    ),
}.items():
    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"{name} debe ser un diccionario no vacío."
        )

for name, value in {
    "EMBEDDING_MODEL_NAME": (
        EMBEDDING_MODEL_NAME
    ),
    "OPENAI_MODEL": OPENAI_MODEL,
    "CHROMA_COLLECTION_NAME": (
        CHROMA_COLLECTION_NAME
    ),
}.items():
    if not value:
        raise ValueError(
            f"{name} no puede estar vacío."
        )

for path in [
    PROJECT_DIR,
    SRC_DIR,
    EXPERIMENT_DIR,
    GROUND_TRUTH_DIR,
    INPUT_PDFS_DIR,
    EXTRACTED_TEXTS_DIR,
    CHUNKS_DIR,
    CHROMA_DIR,
    OUTPUTS_DIR,
    SCIENTIFIC_EXTRACTION_DIR,
    KNOWLEDGE_BASE_DIR,
    THEMATIC_ANALYSIS_DIR,
    OUTLINE_DIR,
    DRAFT_DIR,
    VERIFICATION_TRACEABILITY_DIR,
    EVALUATION_DIR,
    ORCHESTRATOR_DIR,
]:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )
