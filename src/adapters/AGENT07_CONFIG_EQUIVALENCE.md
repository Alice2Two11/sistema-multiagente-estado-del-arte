# Mapa de equivalencia: configuración del Agente 07 (notebook vs. adaptador)

Este documento compara, campo por campo, la celda 7 (rama productiva, no
`FIXTURE_MODE`) de `07_agente_verificador_trazabilidad_LIMPIO.ipynb` contra
`src/adapters/verification_orchestrator_runtime.load_verification_configuration`.

Fuente del notebook citada aquí verbatim desde el `.ipynb` (celda 7),
extraída el mismo día que este documento — cualquier divergencia futura
entre el notebook y esta tabla debe resolverse re-extrayendo la celda, no
editando esta tabla de memoria.

Convención de columnas: **Variable original** (notebook) | **Origen**
(celda/módulo) | **Regla real** (cómo el notebook la calcula) | **Campo del
adaptador** | **Regla del adaptador** | **Evidencia de equivalencia**.

## 1. Modelos

| Variable original | Origen | Regla real (notebook) | Campo del adaptador | Regla del adaptador | Evidencia de equivalencia |
|---|---|---|---|---|---|
| `OPENAI_MODEL` | notebook 00, celda 3: `OPENAI_MODEL = "gpt-4.1-mini"` (constante fija, no interactiva) | `active_experiment.json["openai_model"] = OPENAI_MODEL` (celda 8); `config.py`: `OPENAI_MODEL = str(ACTIVE_EXPERIMENT["openai_model"]).strip()` (celda 9) | `agent07_config["verification_model"]`, `["correction_model"]`, `["reverification_model"]` | `str(_require_active_experiment_key(active, "openai_model")).strip()` — **obligatorio, sin default** (antes tenía `active.get("openai_model", "gpt-4o-mini")`, un default erróneo que ni siquiera coincidía con el real `"gpt-4.1-mini"`; corregido) | **VERIFICADA**: clave `active_experiment.json["openai_model"]` confirmada por lectura directa de las celdas 3/8/9 del notebook 00 real. `config.py` es un passthrough puro (solo `.strip()`); el adaptador reproduce exactamente ese passthrough. Prueba: `test_verification_notebook00_equivalence.py::test_openai_model_equivalence`. |
| `EMBEDDING_MODEL_NAME` | notebook 00, celda 3: `DEFAULT_GENERATION_PROFILE["embedding_model"] = "all-MiniLM-L6-v2"`; celda 8: `active_experiment.json["embedding_model"] = generation_profile["embedding_model"]`; `config.py` celda 9: `EMBEDDING_MODEL_NAME = str(ACTIVE_EXPERIMENT["embedding_model"]).strip()` | `agent07_config["embedding_model"]` | `str(_require_active_experiment_key(active, "embedding_model")).strip()` — obligatorio, sin default; se eliminó la prioridad artificial de `embedding_model_name` (nunca fue una clave real) | **VERIFICADA**: clave real confirmada como `"embedding_model"` (no `"embedding_model_name"`) contra las celdas 3/8/9. `config.py` la exige en `required_keys` y la valida no vacía (`if not value: raise ValueError(f"{name} no puede estar vacío.")`, incluida en el mismo bloque que `OPENAI_MODEL`/`CHROMA_COLLECTION_NAME`) — el adaptador ahora reproduce exactamente ese mismo trato. Pruebas: `test_embedding_model_equivalence`, `test_missing_embedding_model_raises`, `test_empty_embedding_model_raises`, y `test_verification_build_execution_real.py::B1` (confirma que `Agent07ChromaRetriever.embedding_model` recibe el mismo valor). |

## 2. Prompts y versiones

(sin cambios respecto a la versión anterior de este documento — verificados
por identidad de código, no dependen de `active_experiment.json`.)

| Variable original | Origen | Regla real | Campo del adaptador | Regla del adaptador | Evidencia |
|---|---|---|---|---|---|
| `complete_policy["verification_user_prompt_version"]` | `get_verification_input_policy()` (`src/config/verification_policy_config.py`, MISMO módulo en notebook y adaptador) | Constante de la política por defecto, sin overrides posibles para la versión de prompt en sí | `agent07_config["verification_prompt_version"]` | Idéntica: `complete_policy["verification_user_prompt_version"]`, llamando a la MISMA función `get_verification_input_policy()` | **VERIFICADA por identidad de código**: ambos llaman exactamente a la misma función del mismo módulo, sin transformación intermedia. |
| `correction_user_prompt_version`, `reverification_user_prompt_version` | ídem | ídem | `agent07_config["correction_prompt_version"]`, `["reverification_prompt_version"]` | ídem | Mismo nivel de evidencia que arriba — verificada por identidad de código. |

## 3. Políticas de verificación / corrección / reverificación

| Variable original | Origen | Regla real | Campo del adaptador | Regla del adaptador | Evidencia |
|---|---|---|---|---|---|
| `VERIFICATION_POLICY` (= `FIXED_VERIFICATION_POLICY`) | notebook 00, celda 3: diccionario fijo de 15 claves técnicas (`temperature`, `max_chunk_chars`, `top_k_independent_evidence_per_claim`, `allow_automatic_corrections`, etc. — ver el documento fuente citado abajo) | `active_experiment.json["verification_policy"] = FIXED_VERIFICATION_POLICY` (celda 8, sin transformación); `config.py`: `VERIFICATION_POLICY = ACTIVE_EXPERIMENT["verification_policy"]` (celda 9, passthrough puro) | `agent07_config["verification_policy"]` (y también `["correction_policy"]` — ver nota abajo) | `verification_overrides = dict(_require_active_experiment_key(active, "verification_policy"))`; se exige no vacío; se mezcla con `dict.update()` sobre `get_verification_input_policy()` | **VERIFICADA**: clave y contenido exactos confirmados leyendo las celdas 3/8/9 del notebook 00 real (no por convención). Antes el adaptador usaba `active.get("verification_policy", {})` — un default silencioso `{}` que el `config.py` real jamás permitiría (levantaría `ValueError` si la clave faltara). Corregido: ahora es obligatoria, sin default. Prueba: `test_verification_notebook00_equivalence.py::test_verification_policy_equivalence` (compara las 15 claves una por una). |
| `correction_policy` | Notebook: `correction_policy.update(deepcopy(VERIFICATION_POLICY))` — **usa el MISMO `VERIFICATION_POLICY`**, no un `CORRECTION_POLICY` separado (confirmado: no existe ningún `FIXED_CORRECTION_POLICY` en la celda 3) | — | `correction_policy = dict(complete_policy); correction_policy.update(verification_overrides)` | Replica exactamente esa particularidad | **VERIFICADA** — confirmado que el notebook 00 no define una política de corrección separada; el reuso de `VERIFICATION_POLICY` para corrección es intencional y está en el código real, no es una suposición del adaptador. |
| `POST_CORRECTION_RECHECK_POLICY` (= `FIXED_POST_CORRECTION_RECHECK_POLICY`) | notebook 00, celda 3: diccionario fijo de 18 claves | `active_experiment.json["post_correction_recheck_policy"] = FIXED_POST_CORRECTION_RECHECK_POLICY` (celda 8); `config.py`: passthrough puro (celda 9) | `agent07_config["reverification_policy"]` | `reverification_overrides = dict(_require_active_experiment_key(active, "post_correction_recheck_policy"))` | **VERIFICADA**: mismo nivel de evidencia que `verification_policy` — clave y las 18 entradas confirmadas contra la celda 3 real. Antes tenía default silencioso `{}`; corregido a obligatorio. Prueba: `test_post_correction_recheck_policy_equivalence` (18/18 claves comparadas). |

## 4. Presupuestos (budgets)

(sin cambios — verificados por identidad de código, no dependen de `active_experiment.json` ni de `config.py`.)

## 5. Colección Chroma / modelo de embeddings

| Variable original | Origen | Regla real | Campo del adaptador | Regla del adaptador | Evidencia |
|---|---|---|---|---|---|
| `CHROMA_COLLECTION_NAME` | notebook 00, celda 3: `CHROMA_COLLECTION_NAME = "reference_papers_chunks"` (constante fija) | `active_experiment.json["chroma_collection_name"] = CHROMA_COLLECTION_NAME` (celda 8); `config.py`: `CHROMA_COLLECTION_NAME = str(ACTIVE_EXPERIMENT["chroma_collection_name"]).strip()` (celda 9) | `agent07_config["collection_name"]` / `["chroma_collection_name"]` | `collection_name = str(_require_active_experiment_key(active, "chroma_collection_name")).strip()` — obligatorio, sin default | **VERIFICADA**: clave y valor confirmados contra la celda 3/8/9 real. El default anterior (`"reference_papers_chunks"`) coincidía por casualidad con el valor fijo real, pero seguía siendo un default silencioso que `config.py` real no permitiría si la clave faltara — corregido a obligatorio. Prueba: `test_chroma_collection_name_equivalence`. |

## 6-8. (sin cambios respecto a la versión anterior de este documento)

Ver más abajo la sección "Historial de cierre de NO VERIFICADA" para el
resumen de qué se cerró en esta ronda y qué sigue pendiente.

## Resumen de riesgo (actualizado)

**Verificado por identidad de código** (sin dependencia de `active_experiment.json`
ni `config.py`): todos los `*_prompt_version`, todos los `*_budgets`, todos los
`schema_versions`.

**VERIFICADO en esta ronda contra el notebook 00 real** (ya no "inferido por
convención"): `openai_model` (celdas 3/8/9), `chroma_collection_name`
(celdas 3/8/9), `verification_policy` (celdas 3/8/9, 15/15 claves),
`post_correction_recheck_policy` (celdas 3/8/9, 18/18 claves). Los cuatro
son **obligatorios** en `active_experiment.json` — `config.py` real levanta
`ValueError` si falta cualquiera; el adaptador ahora hace lo mismo
(`MissingRequiredActiveExperimentKeyError`), sin ningún default silencioso.
Pruebas: `tests/orchestration/test_verification_notebook00_equivalence.py`
(9 escenarios) + `tests/orchestration/test_verification_build_execution_real.py`
(2 escenarios, atravesando el constructor real completo).

**Cerrado en la ronda 3** (ver "Resumen de riesgo (actualizado — ronda 3,
cierre)" al final del documento): `embedding_model`, `rag_policy`,
`code_root`/`project_root`/`experiment_root` y el resto de
`experiment_paths` (`ORCHESTRATOR_DIR`, `OUTLINE_DIR`,
`VERIFICATION_TRACEABILITY_DIR`, `OUTPUTS_DIR`, `CHROMA_DIR`, `CHUNKS_DIR`,
staging de 07).

**Sigue fuera de alcance, no pedido en ninguna ronda:**
- Todas las demás políticas (`extraction_policy`, `thematic_analysis_policy`,
  etc.) que usan el mismo patrón `active.get("<etapa>_policy", {})` con
  default silencioso en los adaptadores de 02-06 (código preexistente, no
  escrito en esta iteración) probablemente tienen el mismo problema que
  tenían `verification_policy`/`post_correction_recheck_policy` antes de
  esta corrección — no se tocó porque está fuera del alcance de esta
  petición (solo 07), pero se señala como hallazgo.

## 5b. Política RAG (`rag_policy`) — verificada esta ronda

`get_rag_policy()` (`src/rag_policy.py`, notebook 00 celda 11) **NO es un
passthrough** de `active_experiment.json["rag_policy"]`: valida y RESHAPEA
la política antes de que 07 la consuma. El adaptador anterior pasaba el
dict crudo tal cual (`active.get("rag_policy", {})`) — incorrecto tanto en
obligatoriedad (default silencioso `{}`) como en forma (no reproducía la
transformación real). Corregido con `_derive_rag_policy_like_notebook00`.

| Pieza | Origen | Transformación real (`rag_policy.py`, celda 11) | Adaptador | Evidencia |
|---|---|---|---|---|
| clave persistida | notebook 00, celda 3: `FIXED_RAG_POLICY` (10 claves: `exclude_review_sections_from_reference_papers`, `excluded_reference_section_types`, `ground_truth_usage`, 4× `use_ground_truth_for_*`, `retrieval_profiles`, `indexing`, `generation`) | `active_experiment.json["rag_policy"] = FIXED_RAG_POLICY` (celda 8, sin transformación) | `raw_rag_policy = _require_active_experiment_key(active, "rag_policy")` — obligatorio, no vacío | **VERIFICADA**: mismas 10 claves confirmadas contra la celda 3 real. |
| `ground_truth_policy` | — | `rag_policy.py` construye un dict anidado con los 4 `use_ground_truth_for_*`, casteados a `bool`, y **valida** que solo `use_ground_truth_for_evaluation` sea `True` (los otros 3 deben ser `False`, o lanza `ValueError`) | `_derive_rag_policy_like_notebook00` reproduce exactamente esa construcción y esas 2 validaciones | **VERIFICADA**. Prueba: `test_rag_policy_transformation_equivalence`, `test_rag_policy_ground_truth_misuse_raises`. |
| `review_section_types` | `excluded_reference_section_types` (lista) | **renombrada** a `review_section_types` y convertida a `set` → `sorted(...)` (lista ordenada); valida no vacío | reproducido igual | **VERIFICADA**. |
| `review_section_labels_es`, `review_section_patterns`, `rag_allowed_content_policy` | — | **Hardcodeados en `rag_policy.py`**, NO vienen de `active_experiment.json` (7 etiquetas ES, 17 patrones regex, 1 string fijo) | copiados verbatim como `_NOTEBOOK00_REVIEW_SECTION_LABELS_ES`/`_NOTEBOOK00_REVIEW_SECTION_PATTERNS`/`_NOTEBOOK00_RAG_ALLOWED_CONTENT_POLICY` | **VERIFICADA por copia literal** de la celda 11 completa (no por convención). |
| `retrieval_profiles`, `indexing`, `generation` | mismas claves de `FIXED_RAG_POLICY` | passthrough (con `deepcopy`), más validación de estructura (`batch_size` requerido, `temperature`/`answer_max_words` requeridos, rangos) | reproducido igual, mismas validaciones | **VERIFICADA**. Esta es la única parte que el retriever (`Agent07ChromaRetriever`) realmente consume (`retrieval_profiles.default.top_k/fetch_k`), y sobrevive intacta a la transformación — por eso el wiring del retriever no necesitó cambios. |
| `index_batch_size`, `rag_temperature`, `rag_answer_max_words` | — | aplanados desde `indexing`/`generation`, casteados y validados (`>0`, `0≤temp≤2`) | reproducido igual | **VERIFICADA**. |

Pruebas: `test_rag_policy_transformation_equivalence` (compara las 11 claves
del resultado), `test_incomplete_rag_policy_raises` (política sin
`retrieval_profiles` → mismo error), `test_rag_policy_ground_truth_misuse_raises`
(Ground Truth mal configurado → mismo error).

## 6. Rutas de entrada y salida (`experiment_paths`) — actualizado esta ronda

Todas las rutas de esta sección, salvo `code_root`/`project_root`, son
**derivadas de `experiment_dir`** en `config.py` real (celda 9) — NINGUNA
está persistida como clave en `active_experiment.json` (la celda 8 solo
escribe `experiment_dir`/`project_dir`, no rutas de subdirectorios). El
adaptador anterior ofrecía overrides vía `active.get("chroma_dir", ...)`
que nunca existen en el flujo real — eliminados; ahora se derivan siempre
igual que `config.py`, sin indirección.

| Variable original | Origen | Regla real (`config.py`, celda 9) | Campo del adaptador | Regla del adaptador | Evidencia |
|---|---|---|---|---|---|
| `code_root` | `REPO_ROOT` (variable de notebook, celda anterior a la 3, no citada aquí) | ruta del repo clonado en Colab | `experiment_paths["code_root"]` | `Path(__file__).resolve().parents[2]` | **Deliberadamente NO verificada ni validada contra un literal** — `config.py` real hardcodea `PROJECT_DIR = Path("/content/proyecto_estado_arte")` (confirmado, celda 9), es decir, es una ruta de la sesión de Colab, no un contrato semántico (igual que `code_root`). `validate_agent07_orchestrator_compatibility` no la valida a propósito. |
| `project_root` | `PROJECT_DIR = Path("/content/proyecto_estado_arte")` (hardcodeado en `config.py` real, celda 9) | — | `experiment_paths["project_root"]` | `str(root)` = `project_dir` recibido como parámetro | **VERIFICADA como "no forma parte del contrato semántico"**: confirmado que el notebook real la hardcodea a un literal de Colab; el adaptador la parametriza correctamente en su lugar (elección de diseño ya correcta, no un gap). |
| `experiment_root` / `root` | `EXPERIMENT_DIR = PROJECT_DIR / EXPERIMENT_ID` (celda 9) | — | `experiment_paths["experiment_root"]` / `["root"]` | `root / active["active_experiment_id"]` | **VERIFICADA**: misma regla de construcción exacta. |
| `OUTPUTS_DIR` | `EXPERIMENT_DIR / "05_outputs"` (celda 9) | — | (interno, `outputs`) | `experiment_dir / "05_outputs"` | **VERIFICADA**. |
| `ORCHESTRATOR_DIR` → `pipeline_state_path` | `OUTPUTS_DIR / "00_orchestrator_planner"`; `CANONICAL_STATE_PATH = ORCHESTRATOR_DIR / "pipeline_state.json"` (celda 7) | — | `experiment_paths["pipeline_state_path"]` | `outputs / "00_orchestrator_planner" / "pipeline_state.json"` | **VERIFICADA**, byte a byte. Prueba: `test_paths_match_config_py_derivation`. |
| `OUTLINE_DIR` → `outline_paper_mapping_path` | `OUTPUTS_DIR / "04_outline"`; `CANONICAL_OUTLINE_MAPPING_PATH = OUTLINE_DIR / "outline_paper_mapping.csv"` (celda 7) | — | `experiment_paths["outline_paper_mapping_path"]` | `outputs / "04_outline" / "outline_paper_mapping.csv"` | **VERIFICADA**. |
| `VERIFICATION_TRACEABILITY_DIR` → `agent07_output_dir` | `OUTPUTS_DIR / "06_verification_traceability"` (celda 9); `AGENT07_OUTPUT_DIR = VERIFICATION_TRACEABILITY_DIR` (celda 7) | — | `experiment_paths["agent07_output_dir"]` | `outputs / "06_verification_traceability"` | **VERIFICADA**. |
| staging de 07 → `agent07_staging_dir` | `AGENT07_STAGING_DIR = OUTPUTS_DIR / ".agent07_staging"` (celda 7, idéntico en `FIXTURE_MODE` y rama productiva) | — | `experiment_paths["agent07_staging_dir"]` | `outputs / ".agent07_staging"` | **VERIFICADA**. |
| `CHROMA_DIR` → `chroma_dir`/`chroma_manifest_path` | `EXPERIMENT_DIR / "04_chroma_index"` (celda 9, **directo bajo `experiment_dir`, NO bajo `OUTPUTS_DIR`**); `CHROMA_MANIFEST_PATH = CHROMA_DIR / "chroma_index_manifest.json"` (celda 7) | — | `cfg["chroma_dir"]` / `cfg["chroma_manifest_path"]` | `experiment_dir / "04_chroma_index"` / `chroma_dir / "chroma_index_manifest.json"` | **VERIFICADA — bug real corregido esta ronda**: la versión anterior tenía un prefijo espurio `outputs/"01_rag"/...` que no existe en `config.py` real; el `CHROMA_DIR` real cuelga directo de `EXPERIMENT_DIR`, no de `OUTPUTS_DIR`. Corregido. Prueba: `test_paths_match_config_py_derivation`. |
| `CHUNKS_DIR` → `chunks_dir`/`chunks_manifest_path` | `EXPERIMENT_DIR / "03_chunks"` (celda 9); `CHUNKS_MANIFEST_PATH = CHUNKS_DIR / "chunks_clean_for_rag.jsonl"` (celda 7) | — | `cfg["chunks_dir"]` / `["chunks_manifest_path"]` | `experiment_dir / "03_chunks"` / `chunks_dir / "chunks_clean_for_rag.jsonl"` | **VERIFICADA** — esta ya coincidía antes de esta ronda; confirmado formalmente ahora con cita exacta. |

## 7. Esquemas (`schema_versions`)

| Variable original | Origen | Regla real | Campo del adaptador | Regla del adaptador | Evidencia |
|---|---|---|---|---|---|
| `PROVISIONAL_BUNDLE_FINGERPRINT_VERSION` | `src/config/verification_policy_config.py` (mismo módulo) | Constante | `schema_versions["provisional_bundle"]` | Misma constante, mismo import | **VERIFICADA por identidad de código.** |
| `RESOLUTION_FP_VERSION` | `src/tools/verification/resolution.py` (mismo módulo) | Constante | `schema_versions["multi_proposal_resolution"]` | Misma constante, mismo import | Verificada por identidad de código. |
| `AGENT07_RUNTIME_METRICS_VERSION` | `src/adapters/verification_runtime.py` (mismo módulo) | Constante | `schema_versions["runtime_metrics"]` | Misma constante, mismo import | Verificada por identidad de código. |

## 8. Configuración del experimento activo (`active_experiment_config`)

Esta pieza NO forma parte de `agent07_config`: el notebook la construye
aparte, solo para pasarla a `validate_agent07_experiment_compatibility`.

| Variable original | Origen | Regla real | Campo propuesto | Regla propuesta | Evidencia |
|---|---|---|---|---|---|
| `active_experiment_config["active_experiment_id"]` | `EXPERIMENT_ID` (`config.py`) | — | ídem | `active["active_experiment_id"]` (de `active_experiment.json`, la misma fuente que ya usan 02-06) | Fuente distinta de la del notebook (`config.py` vs `active_experiment.json`) pero **debieran contener el mismo valor**, porque notebook 00 genera `config.py` a partir de `active_experiment.json`. No verificado contra una generación real. |
| `active_experiment_config["rag_policy"]` | `get_rag_policy()` (`rag_policy.py`, generado por notebook 00) | — | ídem | `active.get("rag_policy", {})` | Misma relación que la fila anterior — 02-06 ya usan `active.get("rag_policy", {})` como la fuente canónica. |
| resto de campos (`embedding_model`, `openai_model`, `chroma_collection_name`, `verification_policy`, `verification_prompt_version`, `verification_budgets`) | `config.py` / derivados de `agent07_config` ya construido | — | ídem | derivados de las secciones 1-4 de este documento | Hereda el mismo nivel de evidencia (verificado o no) que su campo de origen correspondiente arriba. |

## Resumen de riesgo (actualizado — ronda 3, cierre)

**Verificado por identidad de código**: todos los `*_prompt_version`, todos
los `*_budgets`, todos los `schema_versions`.

**Verificado contra el notebook 00 real, ronda 2**: `openai_model`,
`chroma_collection_name`, `verification_policy`, `post_correction_recheck_policy`.

**Verificado contra el notebook 00 real, ronda 3 (esta)**: `embedding_model`
(clave real confirmada, prioridad artificial de `embedding_model_name`
eliminada), `rag_policy` (transformación completa de `get_rag_policy()`
reproducida, no un passthrough), y las rutas derivadas de `experiment_dir`
(`ORCHESTRATOR_DIR`, `OUTLINE_DIR`, `VERIFICATION_TRACEABILITY_DIR`,
`OUTPUTS_DIR`, `CHROMA_DIR`, `CHUNKS_DIR`, staging de 07) — con un bug real
corregido (`chroma_dir`/`chroma_manifest_path` tenían un prefijo espurio
`05_outputs/01_rag` que no existe en `config.py` real).

**Deliberadamente sin verificar, por diseño** (no forman parte del contrato
semántico, confirmado contra `config.py` real que las hardcodea a literales
de Colab): `code_root`, `project_root`. El adaptador los parametriza
correctamente en su lugar; esto ya es la solución, no un gap pendiente.

**No quedan campos pendientes de los 7 bajo verificación en esta ronda**
(los 4 de la ronda 2 + `embedding_model`/`rag_policy`/rutas de esta ronda).
Total de pruebas de equivalencia: 16 (`test_verification_notebook00_equivalence.py`)
+ 2 (`test_verification_build_execution_real.py`, atravesando el
constructor completo) = 18, todas contra lógica real de notebook 00
extraída verbatim, no contra valores inventados.
