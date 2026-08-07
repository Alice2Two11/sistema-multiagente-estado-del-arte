# Inventario técnico exhaustivo de la etapa 08 (evaluación experimental)

Reemplaza el inventario superficial de una iteración anterior. Fuente:
`08_evaluacion_12_CITAS_Y_NUMEROS_CORREGIDOS.ipynb`, 26 celdas, ~115 KB,
leídas completas celda por celda (no por muestreo) para este documento.
**No se escribió ningún `StageSpec` ni se portó código en esta iteración**
— es inventario puro, por instrucción explícita.

---

## 1. Entradas reales

| Entrada | Ruta exacta | Cómo se resuelve | Obligatoria |
|---|---|---|---|
| Estado del arte generado (evaluado) | **Ruta activa (07 directo)**: sintetizada por `resolve_agent08_upstream_input` a partir del bundle committed de 07. **Ruta histórica, no vigente (07C)**: `EVALUATION_READY_JSON_PATH`/`_MD_PATH` = `DIR_VERIFY/"verified_state_of_art_EVALUATION_READY.{json,md}"`, salida real de 07C | `resolve_agent08_upstream_input(agent07_directory=DIR_VERIFY, agent07c_directory=DIR_VERIFY, draft_json_path=DRAFT_JSON_PATH, draft_markdown_path=DRAFT_MD_PATH)` (celda 5, `src/adapters/evaluation_upstream.py`, YA EXISTE) — decide la rama según si existe `post_correction_recheck_manifest.json`; como 07C nunca corre en el orquestador activo, siempre resuelve a la rama directa | Sí |
| Draft committed de 06 (para hashes de integridad) | `pipeline_state.json["artifacts"]["state_of_art_draft.json"/".md"]` | Lectura directa de `pipeline_state.json` + verificación de hash (celda 3) — **sin StateStore**, JSON crudo | Sí |
| Salida verificada/trazabilidad de 07 | **Ruta activa (07 directo, la única vigente)**: `DIR_VERIFY/"provisional_verification_traceability_bundle.json"` (celda 9), transformado por 08 en `df_traceability`/`df_recheck_report`/`df_numeric_recheck` sintéticos. **Ruta histórica, no vigente (07C)**: `RECHECK_TRACEABILITY_CSV_PATH`/`RECHECK_NUMERIC_CSV_PATH`/`RECHECK_REPORT_CSV_PATH`/`RECHECK_VALIDATION_REPORT_PATH`/`RECHECK_MANIFEST_PATH`, todos bajo `DIR_VERIFY/"post_correction_*"` — artefactos que solo existen si 07C corrió | Bifurca por `source_stage` (`"AGENT07C"` vs. resto); en el orquestador activo `source_stage` nunca es `"AGENT07C"` | Sí (rama directa siempre, en el orquestador activo) |
| Ground Truth | `GROUND_TRUTH_DIR = config_module.GROUND_TRUTH_DIR` = `EXPERIMENT_DIR/"00_ground_truth"`. Preferencia: `ground_truth_literature_review.txt` (preextraído) → `ground_truth_full_text.txt`/`ground_truth_text.txt` → único PDF en el directorio (`extract_pdf_text` vía `fitz`) | `load_ground_truth_full_text()` + `extract_gt_literature_review()` (celda 13) — falla si hay 0 o >1 PDF sin TXT preextraído | Sí |
| Tabla comparativa (cuantitativa, de 03B) | **No se referencia en ningún punto de las 26 celdas** — 08 no consume `quantitative_comparative_table.csv` ni artefactos de 03B | — | No aplica a 08 |
| Chunks autorizados | `CHUNKS_CSV_PATH = CHUNKS_DIR/"chunks_clean_for_rag.csv"` | Se valida que tenga `source_filename`/`chunk_id`/`text` y que NINGUNA fila sea de Ground Truth (regex `ground[_\s-]*truth\|gt_`) — celda 11 | Sí |
| Configuración y políticas | `config.py`: `EVALUATION_POLICY` (23 claves, ver §4); `experiment_config.py`: `EXPERIMENT_PROFILE`; `generation_config.py`: `GENERATION_PROFILE` | `import config as config_module` + `importlib.reload` (celda 1) | Sí, las 3 |
| Credencial OpenAI | `.runtime_secrets/openai_api_key.txt` o `os.environ["OPENAI_API_KEY"]` | Mismo patrón de bloqueo de Colab Secrets que 07/08 comparten (celda 1) | Sí (solo si `RUN_LLM_JUDGE=True`) |

## 2. Métricas implementadas (15 finales, más las intermedias)

| # | Métrica final | Bloque | Librería/método | Detalle |
|---|---|---|---|---|
| 1 | `rougeL_fmeasure` (+ precision/recall) | Automáticas | `rouge_score.rouge_scorer.RougeScorer(["rougeL"])` | Sobre texto plano generado vs. GT, con traducción opcional si difieren idiomas (`translate_for_rouge_if_language_differs`) |
| 2 | `bertscore_f1` | Automáticas | `bert_score.score()`, modelo `EVALUATION_POLICY["bertscore_model"]` (típ. `bert-base-multilingual-cased`) | Por oraciones, máx. `max_bertscore_pairs`; se registra el valor medio y también un máximo por alineación |
| 3 | `semantic_f1` | Automáticas | `sentence_transformers.SentenceTransformer(EVALUATION_EMBEDDING_MODEL)` + `sklearn.metrics.pairwise.cosine_similarity` | Chunking por oraciones (`chunk_text_by_sentences`, `SEMANTIC_CHUNK_CHARS`/`_OVERLAP_CHARS`), alineación bidireccional |
| 4 | `global_semantic_similarity` | Automáticas | mismo embedding, a nivel documento completo | `cosine_similarity_mean_document_embeddings` |
| 5-9 (5) | Puntajes LLM Judge (rúbrica, 1-5) | LLM Judge | `ChatOpenAI` vía `get_llm()`, prompt versionado (`LLM_JUDGE_PROMPT_VERSION`) | Un único `invoke()` (con reintentos hasta `llm_judge_max_attempts`), JSON validado por `validate_judge_result`; tiene su propio cache por fingerprint (`judge_fingerprint`/`old_judge_manifest`) |
| 10 | `factual_precision` | Factuales | derivado de `df_final_claim_audit` | claims soportados / claims totales |
| 11 | `hallucination_rate` | Factuales | ídem | claims no soportados / total |
| 12 | `evidence_coverage` | Factuales | ídem | cobertura de evidencia citada |
| 13 | `traceability_text_coverage` | Factuales | ídem | cobertura de texto trazable |
| 14 | `citation_error_rate` | Factuales (citas) | regex `citation_pattern` sobre `[fuente\|chunk]`, validado contra `valid_source_chunk_pairs` | recalcula citas del texto FINAL evaluado, no reusa las de 06/07 |
| 15 | `numeric_error_rate` | Factuales (numérico) | `numeric_pattern` + `normalize_numeric_token`/`numeric_search_variants`, contra chunks citados | recalcula sobre el texto final |

`factual_consistency_ok` = `factual_precision==1.0 and hallucination_rate==0.0 and evidence_coverage==1.0 and traceability_text_coverage==1.0` (celda 19, umbral binario estricto, no configurable).

## 3. Salidas reales — corregido esta ronda

**Corrección de un error real del inventario anterior**: había incluido
`numeric_hallucination_check.csv` como una de las 15 salidas obligatorias
de 08. Es INCORRECTO — verificado por grep exhaustivo contra las 26 celdas:
ese nombre de archivo **no aparece en ningún punto del notebook 08**. Es un
artefacto de la etapa **06** (`AGENT06_REQUIRED_ARTIFACTS` en
`src/adapters/agent06_verification_handoff.py`), que 07 consume — no algo
que 08 produzca ni consuma. Lo elimino de esta lista.

La lista exacta de 15 archivos obligatorios es la variable
`required_outputs` construida literalmente en la celda 23 (el bloque que
levanta `RuntimeError` si `missing_outputs` no está vacío):

Todos bajo `DIR_EVALUATION = OUTPUTS_DIR/"07_evaluation"`:

| # | Archivo | Tipo | Contenido |
|---|---|---|---|
| 1 | `automatic_metrics.csv` | CSV | ROUGE-L/BERTScore/semánticas |
| 2 | `semantic_chunk_alignment.csv` | CSV | alineación de chunks semánticos |
| 3 | `bertscore_chunk_alignment.csv` | CSV | alineación BERTScore por oración |
| 4 | `factual_metrics.csv` | CSV | las 6 métricas factuales |
| 5 | `final_citation_check.csv` | CSV | auditoría de citas recalculada |
| 6 | `final_claim_audit.csv` | CSV | auditoría de claims final |
| 7 | `llm_judge_evaluation.json` | JSON | resultado crudo del LLM Judge |
| 8 | `llm_judge_scores.csv` | CSV | puntajes 1-5 por criterio |
| 9 | `corpus_gap_suggestions.csv` | CSV | brechas temáticas frente al GT |
| 10 | `corpus_gap_suggestions.md` | MD | mismas brechas, en Markdown (entrada separada en `required_outputs`, no la misma que la 9) |
| 11 | `final_selected_metrics.csv` | CSV | las 15 métricas finales unificadas |
| 12 | `evaluation_summary.json` | JSON | resumen ejecutable (ver §6) |
| 13 | `final_evaluation_report.md` | MD | reporte legible completo |
| 14 | `evaluation_validation_report.json` | JSON | `validation_ok`/`errors`/`warnings` |
| 15 | `evaluation_manifest.json` | JSON | manifiesto final (fingerprint, inputs, outputs, `safety_policy`) |

**Sobre el archivo numérico que SÍ toca 08** (aclaración pedida
explícitamente — no es ninguna de las 3 opciones tal cual planteadas, es
más específico): `RECHECK_NUMERIC_CSV_PATH` (nombre real de variable,
NUNCA literalmente `"numeric_hallucination_check.csv"`) sigue una ruta
distinta según el origen:
- **Ruta activa (07 directo, la única vigente en el orquestador)**: en la
  celda 9, `RECHECK_NUMERIC_CSV_PATH` se fija a
  `DIR_EVALUATION/"agent08_upstream_numeric_check.csv"` y 08 mismo lo
  **crea** ahí (no es una entrada heredada: lo sintetiza desde
  `upstream.numeric_check_rows`). Después, en la celda 19, 08
  **sobrescribe ese mismo archivo** con una versión recalculada contra el
  texto final evaluado. Es decir: en la ruta activa, es un archivo que 08
  escribe dos veces (crea y luego recalcula), no una entrada externa.
- **Ruta histórica (07C, excluida del orquestador activo)**: apuntaría a
  `DIR_VERIFY/"post_correction_numeric_check.csv"`, una salida real de 07C
  que 08 solo leería. Esta rama no aplica a nuestro flujo — no se debe
  diseñar el runtime nuevo asumiendo su existencia.
- En ningún caso `RECHECK_NUMERIC_CSV_PATH` aparece en la lista de 15
  `required_outputs` de la celda 23 — no es parte de la auditoría final de
  completitud, es un archivo de trabajo intermedio.

Auxiliares no listados como obligatorios pero escritos: `generated_evaluation_ready_text.txt`, `ground_truth_literature_review_text.txt`, `ground_truth_extraction_metadata.json`, `generated_text_translated_to_ground_truth_language.txt` + `translation_manifest.json` (si aplica traducción), `llm_judge_manifest.json`, `raw_llm_judge_outputs/` (directorio).

## 4. Dependencias

**Librerías** (no usadas por 02-07): `rouge_score`, `bert_score`, `sentence_transformers`, `sklearn.metrics.pairwise`, `langdetect`, `fitz` (ya usado en 01/02, reutilizable). Comunes con 07: `langchain_openai`/`langchain_core.messages`, `pandas`, `numpy`.

**Modelos:** `OPENAI_MODEL` (mismo que 07, vía `config.py`) para LLM Judge; `EVALUATION_POLICY["evaluation_embedding_model"]` (default real: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, **distinto** del embedding de RAG `all-MiniLM-L6-v2`); `EVALUATION_POLICY["bertscore_model"]` (`bert-base-multilingual-cased`).

**Prompts:** uno solo, `build_judge_prompt` (celda 21), versión `LLM_JUDGE_PROMPT_VERSION = "v5_rubric_reference_comparison_strict_json"` — hardcodeado en el notebook, no en `active_experiment.json`.

**Llamadas a OpenAI:** una sola invocación real por corrida (`llm_judge.invoke(...)`, con reintentos internos hasta `llm_judge_max_attempts`, cacheada por fingerprint entre corridas).

**Ya existe en `src/`:** únicamente `src/adapters/evaluation_upstream.py` (resolución de qué input usar, 07 vs 07C — ya integrado en `src/`, sin cambios necesarios).

**Vive exclusivamente en el notebook** (todo lo demás): `extract_gt_literature_review`, `load_ground_truth_full_text`, `extract_pdf_text`, `normalize_pdf_text`, `find_headings`, `build_heading_pattern`, `split_sentences`, `chunk_text_by_sentences`, `normalize_claim_text`, `normalize_content_text`, `normalize_numeric_token`, `numeric_search_variants`, `strip_internal_citations`, `detect_language_code`, `translate_text_to_language`, `build_judge_prompt`, `validate_judge_result`, `metric_value`, `stable_hash_dict`, `backup_evaluation_outputs`, `reset_evaluation_outputs`, `balanced_excerpt`, `evenly_spaced_items`, más TODO el cálculo de las 15 métricas y la construcción de los 15 archivos de salida.

## 5. Bloques funcionales reales (por celda, sin inventar arquitectura)

1. **Bloqueo de credenciales + carga de config** (celda 1): `config.py`/`experiment_config.py`/`generation_config.py`, validación de `EVALUATION_POLICY` (23 claves obligatorias), definición de las ~25 rutas de `DIR_EVALUATION`.
2. **Resolución del draft committed de 06** (celdas 2-3): lectura de `pipeline_state.json`, verificación de hashes.
3. **Resolución de la ruta upstream (07 vs 07C)** (celdas 4-5): `resolve_agent08_upstream_input` — ya en `src/`.
4. **Carga y validación del texto a evaluar** (celdas 6-7): secciones no vacías, texto plano concatenado.
5. **Normalización de trazabilidad** (celdas 8-9): unifica esquema 07C/07-directo en `df_traceability`/`df_recheck_report`/`df_numeric_recheck`.
6. **Validación de chunks autorizados + aislamiento de Ground Truth** (celdas 10-11): ver §7.
7. **Preparación del Ground Truth** (celdas 12-13): extracción de la sección de revisión de literatura (heurística de encabezados bilingüe), validación de longitud mínima, verificación de que GT ≠ texto generado.
8. **Fingerprint, decisión de rebuild, backup, preprocesamiento** (celdas 14-15): ver §6 — es la pieza más cercana a un PREPARE/freshness-check ya existente.
9. **Métricas léxicas y semánticas (automáticas)** (celdas 16-17): ROUGE-L, BERTScore, similitud semántica.
10. **Evaluación factual** (celdas 18-19): auditoría de claims, citas, valores numéricos.
11. **LLM Judge** (celdas 20-21): única llamada a LLM, con cache propio.
12. **Agregación, brechas de corpus y manifiesto final** (celdas 22-23): las 15 métricas unificadas, `evaluation_summary.json`, `evaluation_validation_report.json`, `evaluation_manifest.json`, `final_evaluation_report.md`.
13. **Presentación** (celdas 24-25): solo `display()` en el propio notebook — no produce artefactos.

Esta división por celdas coincide con los encabezados markdown reales del
notebook. "Métricas léxicas" y "métricas semánticas" están fusionadas en un
solo bloque real ("métricas automáticas") — no las separo artificialmente.
"Evaluación factual", "LLM Judge" y "agregación" sí coinciden con bloques
reales; "persistencia" no es un paso aparte — cada sección escribe sus
propios archivos inline.

## 6. Propuesta de contrato transaccional para 08

08 ya tiene, de forma NO integrada con `StateStore`, un mecanismo de
freshness-check propio (celda 15: `evaluation_fingerprint` vs.
`previous_evaluation_manifest["fingerprint"]` → `SHOULD_REBUILD_EVALUATION`)
y un mecanismo de éxito/fracaso propio (celda 23:
`evaluation_validation_ok` + `validation_errors`/`validation_warnings`).
La propuesta reutiliza ambos en vez de inventar algo nuevo:

- **Fingerprints**: `evaluation_signature` (celda 15) ya es, en esencia, un
  `StageFingerprints.input` — depende de `EVALUATION_POLICY`, `OPENAI_MODEL`,
  hashes de `EVALUATION_READY_JSON/MD`, hashes de trazabilidad/numérico
  upstream, hash del Ground Truth. Se puede envolver directamente con
  `build_stage_fingerprints(input_data=evaluation_signature, config_data=EVALUATION_POLICY, dependencies_data={...hashes de upstream...})` sin rediseñar nada.
- **PREPARE**: equivalente a la construcción de `evaluation_signature` +
  comparación con el manifiesto anterior (celda 15) — ya existe como lógica,
  solo falta llamarla desde `store.prepare_execution(target_stage="08_evaluacion_experimental", ...)` antes de ejecutar.
- **EXECUTE**: el cuerpo de las celdas 16-23 (métricas + LLM Judge +
  agregación), sin cambios de lógica.
- **Persistencia + COMMIT**: los 15 archivos ya se escriben tal cual; falta
  envolver el resultado en un `AgentResult` (ver abajo) y llamar a
  `store.persist_agent_result`/`store.commit_execution`, igual que hacen
  05/06/07.
- **RESUME**: la comparación de fingerprint (celda 15) ya decide
  `outputs_are_current` vs. debe-reconstruirse — es el mismo rol que
  `resolve_*_resume` cumple en 05/06, solo que hoy vive fuera de
  `StateStore`.
- **`AgentResult` equivalente — corregido esta ronda**: el mapeo anterior
  fusionaba dos casos que el notebook real trata con severidad DISTINTA.
  Se documentan por separado, sin convertir automáticamente el caso
  bloqueante en un resultado comprometido normalmente:

  1. **Fallo de ejecución** (excepción técnica antes de completar el
     cálculo — `FileNotFoundError` de Ground Truth, `ValueError` de
     política incompleta, etc.): `execution_status=FAILED`,
     `quality_status=REJECTED`, `error` con el mensaje real,
     `requested_transition=HALT_STAGE`. Mismo patrón que usan 02-07 para
     fallos de preparación (`_runtime_failure_result` y análogos).

  2. **Evaluación terminada pero inválida, con
     `FAIL_ON_INVALID_EVALUATION=True`** (el valor real por defecto —
     `FIXED_EVALUATION_POLICY["fail_on_invalid_evaluation"]=True` en
     notebook 00): confirmado por lectura exacta de la celda 23 que el
     `raise ValueError(...)` ocurre DESPUÉS de escribir
     `evaluation_summary.json`/`evaluation_validation_report.json`, pero
     ANTES de escribir `final_evaluation_report.md`/
     `evaluation_manifest.json` y antes de que se ejecute siquiera el
     chequeo de `missing_outputs` — es decir, en este caso el notebook real
     **nunca llega a producir el conjunto completo de 15 salidas**. Por
     eso este caso NO debe mapearse a un `AgentResult` `COMPLETED`+
     `REJECTED` comprometido con normalidad: eso implicaría que el
     pipeline podría razonar sobre un resultado "terminado" que en
     realidad quedó a medias. Propuesta: tratarlo con la MISMA severidad
     que el caso 1 (`execution_status=FAILED`,
     `requested_transition=HALT_STAGE`), con `error` listando los
     `validation_errors` que motivaron el corte — preservando que el
     notebook real interrumpe el pipeline, no lo deja avanzar
     silenciosamente. **Esto es una propuesta que requiere tu
     confirmación explícita antes de implementarse** — es la parte del
     mapeo con mayor margen de interpretación de todo este documento.
  3. **Evaluación terminada pero inválida, con
     `FAIL_ON_INVALID_EVALUATION=False`** (posible si el usuario
     sobreescribe la política — el notebook no lo prohíbe, solo
     `FIXED_EVALUATION_POLICY` fija el default en `True`): en este caso,
     confirmado por la misma lectura de la celda 23, el notebook SÍ
     continúa y escribe las 15 salidas completas. Este caso, a diferencia
     del anterior, sí corresponde a un `AgentResult`
     `execution_status=COMPLETED`, `quality_status=REJECTED` comprometido
     con normalidad — el pipeline tiene un resultado completo y trazable
     que no pasó la validación de calidad, análogo a `REJECTED` en otras
     etapas.
  4. **Evaluación válida**: `execution_status=COMPLETED`;
     `quality_status=APPROVED_WITH_WARNINGS` si `validation_warnings` no
     está vacío (caso real ya existente:
     `"upstream_partial_factual_consistency_not_approved"`);
     `quality_status=APPROVED` si no hay advertencias.

  Resto del mapeo, sin cambios:
  - `requested_transition`: dado que 08 es la última etapa de
    `CANONICAL_STAGE_ORDER`, `ADVANCE` con `target_stage=None` (fin del
    pipeline) si `COMPLETED`; `HALT_STAGE` si `FAILED` — mismo patrón que
    07 real (nunca `RETURN`, consistente con la decisión ya tomada de no
    activar el ciclo correctivo).
  - `output_artifacts`: los 15 archivos de §3, con `ArtifactReference`
    (path + sha256) — solo cuando existen (casos 3 y 4; en el caso 2 no
    existen todos).
  - `quality_metrics`: las 15 métricas de §2 tal cual, sin transformación
    (cuando existen).
  - `warnings`: mapeo directo de `validation_warnings`.
  - `error`: en los casos 1 y 2, el mensaje de la excepción real
    (`ValueError`/`RuntimeError`/`FileNotFoundError`) o, para el caso 2
    específicamente, los `validation_errors` que motivaron el corte.

Este mapeo es mecánico en los casos 3 y 4 porque 08 YA calcula todo lo que
un `AgentResult` necesita. El caso 2 es la única pieza de este documento
que no es una transcripción directa del comportamiento real sino una
decisión de diseño propuesta — señalada como tal.


## 7. Ground Truth — uso exclusivo en evaluación (verificado)

Tres invariantes independientes, todas confirmadas por código real:

1. **A nivel de política** (`rag_policy.py`, notebook 00 celda 11 — ya
   verificado y probado en la ronda anterior de equivalencia de 07):
   `use_ground_truth_for_generation`/`_for_rag`/`_for_verification` deben
   ser `False`; solo `_for_evaluation` puede ser `True`. `get_rag_policy()`
   levanta `ValueError` si se viola. Esto rige 02-07, no 08.
2. **A nivel de datos, dentro de 08** (celda 11): valida que
   `chunks_clean_for_rag.csv` (el corpus que 02-07 SÍ usan) no contenga
   ninguna fila cuyo `source_filename` matchee `ground[_\s-]*truth|gt_` —
   detección activa, no solo una regla de política.
3. **A nivel físico**: el Ground Truth vive en `GROUND_TRUTH_DIR =
   EXPERIMENT_DIR/"00_ground_truth"`, un directorio separado de
   `01_input_references_pdfs`/`03_chunks`/`04_chroma_index` que usan 01-07.

Conclusión: el aislamiento está verificado en tres capas independientes
(política declarada, dato indexado, ubicación física), no es una promesa
de un solo lugar.

## 8. Riesgos de equivalencia identificados (para la fase de migración)

- **LLM Judge no determinista**: una sola llamada real a OpenAI por corrida
  (cacheada, pero la primera vez es real). Cualquier prueba de
  caracterización que "capture el comportamiento actual" del notebook para
  esta pieza necesariamente usará un doble — no hay forma de capturar
  comportamiento no determinista como oráculo fijo.
- **`detect_language_code`** depende de `langdetect`, que internamente usa
  un `DetectorFactory.seed = 0` fijado por el notebook — reproducible, pero
  hay que preservar esa fijación de semilla al portar.
- **Extracción de Ground Truth por heurística de encabezados** (30+ alias
  ES/EN, regex de patrones de numeración) es la lógica más frágil y menos
  determinista frente a variaciones de formato de PDF — alto riesgo de
  divergencia sutil si se reimplementa en vez de portar literal.
- **`EVALUATION_POLICY` real no verificada contra `active_experiment.json`
  real del usuario** — mismo tipo de riesgo que ya se cerró para
  `verification_policy`/`rag_policy` de 07; aquí sigue abierto hasta
  verificarlo campo por campo (fuera de alcance de este inventario, ver
  plan incremental).
- **`factual_consistency_ok` es un umbral binario estricto** (los 4
  componentes deben ser exactamente 1.0/0.0) — cualquier redondeo o
  diferencia de tipo de dato al portar podría cambiar silenciosamente el
  resultado de `APPROVED` a `REJECTED` o viceversa.

## Plan incremental de migración (propuesta, no ejecutada)

1. **Bloque 1** (bajo riesgo): normalización de texto pura —
   `normalize_content_text`, `strip_internal_citations`,
   `normalize_claim_text`, `normalize_numeric_token`,
   `numeric_search_variants` → `src/tools/evaluation/text_normalization.py`.
   Sin dependencias externas, fácil de caracterizar.
2. **Bloque 2**: extracción de Ground Truth (heurística de encabezados) →
   `src/adapters/evaluation_ground_truth.py`. Mayor riesgo (ver §8);
   requiere el mayor número de pruebas de caracterización con PDFs/TXT
   sintéticos variados.
3. **Bloque 3**: métricas automáticas (ROUGE-L/BERTScore/semántica) →
   `src/tools/evaluation/automatic_metrics.py`. Requiere `sentence-transformers`/`bert_score` reales o dobles deterministas para pruebas.
4. **Bloque 4**: métricas factuales/citas/numéricas →
   `src/tools/evaluation/factual_checks.py`.
5. **Bloque 5**: LLM Judge → `src/tools/evaluation/llm_judge.py`. Requiere
   doble de LLM para pruebas, igual que se hizo con 07.
6. **Bloque 6**: orquestación interna + fingerprint/manifiesto/`AgentResult`
   → `src/adapters/evaluation_runtime.py`, siguiendo el mapeo de §6.
7. **Bloque 7**: `StageSpec` de 08 en el orquestador, mismo patrón que 07
   (probablemente `custom_run`, dado que 08 no tiene el ciclo
   PREPARE/EXECUTE/COMMIT de 3 funciones separadas que sí tiene 07 — habría
   que decidir si se construye ese desglose o se usa el patrón genérico
   `runtime_transaction` de 02-06).

Cada bloque se entrega con sus propias pruebas de caracterización antes de
pasar al siguiente — no se migra todo de una vez.

## A3 — Mapa compacto de `evaluation_policy` (24 claves, verificado)

Fuente real: `FIXED_EVALUATION_POLICY` (notebook 00, celda 3) → escrita
verbatim en `active_experiment.json["evaluation_policy"]` (celda 8, sin
transformación) → `config.py` (celda 9) la reexpone como
`EVALUATION_POLICY`, validando `required_evaluation_policy_keys` (24
nombres, idénticos a `FIXED_EVALUATION_POLICY.keys()` — confirmado, no hay
ninguna clave extra ni faltante en ningún lado) con
`ValueError("EVALUATION_POLICY está incompleta. Faltan: [...]")` si falta
alguna — **sin default silencioso para ninguna de las 24**.

| Clave | Default real (notebook 00) | Consumida por (módulo ya migrado) |
|---|---|---|
| `auto_rebuild` | `True` | `evaluation_fingerprint.resolve_rebuild_decision` |
| `force_rebuild` | `False` | `evaluation_fingerprint.resolve_rebuild_decision` |
| `require_07c_evaluation_ready` | `True` | validación de política en config.py real (no consumida por ningún módulo migrado — 07C excluido) |
| `require_ground_truth_literature_review` | `True` | no consumida directamente por ningún módulo migrado (implícito en `resolve_ground_truth_comparable_text`, que siempre exige la sección) |
| `allow_full_ground_truth_fallback` | `False` | **NO migrada** — ver Riesgos, este flag no se usa en ninguna función de `ground_truth.py` (posible desviación, ver abajo) |
| `require_explicit_ground_truth_end_heading` | `True` | `ground_truth.resolve_ground_truth_comparable_text` |
| `minimum_generated_words` | `100` | `evaluation_pipeline.run_evaluation_pipeline` |
| `minimum_ground_truth_words` | `150` | `ground_truth.resolve_ground_truth_comparable_text` |
| `translation_temperature` | `0.0` | consumida por el llamador al construir `translation_llm_factory` (fuera de los módulos puros) |
| `judge_temperature` | `0.0` | consumida por el llamador al construir `judge_llm_factory` |
| `max_translation_chars_per_chunk` | `5000` | `translation.translate_text_to_language` (vía `evaluation_pipeline`) |
| `semantic_chunk_chars` | `1200` | `semantic_similarity.build_semantic_chunks` |
| `semantic_chunk_overlap_chars` | `150` | `semantic_similarity.build_semantic_chunks` |
| `max_semantic_chunks_per_text` | `80` | `semantic_similarity.build_semantic_chunks` |
| `evaluation_embedding_model` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `semantic_similarity.build_embedding_model` |
| `bertscore_model` | `bert-base-multilingual-cased` | `bertscore.run_bertscore` |
| `max_bertscore_pairs` | `40` | `bertscore.select_bertscore_pair_indices` |
| `translate_for_rouge_if_language_differs` | `True` | `translation.resolve_generated_text_for_rouge` |
| `llm_judge_max_generated_chars` | `18000` | `llm_judge.run_llm_judge` (vía `balanced_excerpt`) |
| `llm_judge_max_ground_truth_chars` | `18000` | `llm_judge.run_llm_judge` |
| `llm_judge_max_attempts` | `3` | `llm_judge.run_llm_judge` |
| `run_llm_judge` | `True` | `evaluation_pipeline.run_evaluation_pipeline` (bloqueante si `False`, no un salto silencioso) |
| `fail_on_invalid_evaluation` | `True` | `final_validation.resolve_final_validation_gate` |
| `create_corpus_gap_suggestions` | `True` | `final_report.build_corpus_gap_rows` |

**Una desviación real corregida en esta ronda, una aclarada como vestigial
del propio notebook (no un hueco de esta migración):**

1. `run_llm_judge`: **corregido**. El notebook real (celda 21) no la usa
   para "saltar" el Judge — es un `ValueError` bloqueante
   (`"La política de evaluación desactivó el LLM Judge, pero esta tesis
   requiere la rúbrica cualitativa."`) si la política lo desactiva.
   `run_evaluation_pipeline` ahora reproduce exactamente ese chequeo, antes
   de invocar el Judge.
2. `allow_full_ground_truth_fallback`: verificado por grep exhaustivo de
   la celda 13 completa (ya leída íntegra en el Bloque 2) — esta clave
   **no aparece en ningún punto** de la lógica real de extracción del
   Ground Truth. Es una clave vestigial del propio `EVALUATION_POLICY`
   (validada como presente por `config.py`, pero sin ningún consumidor en
   el notebook real) — no es un hueco de esta migración, es un hallazgo
   sobre el notebook original.
