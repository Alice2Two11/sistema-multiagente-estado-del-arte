# Sistema multiagente para generación y evaluación automatizada de estados del arte

## 1. Nombre y propósito del proyecto

Este repositorio contiene el código fuente reutilizable y las pruebas
automatizadas de un **sistema multiagente basado en modelos de lenguaje
grandes (LLM)** que genera, verifica y evalúa automáticamente **estados del
arte** (revisiones de literatura científica) a partir de un corpus de
artículos proporcionado por el usuario.

Es el soporte de código de una tesis de maestría. Este `README.md` describe
el **estado real y verificado del sistema completo** — el repositorio de
código (`src/`, `tests/`) y los notebooks operativos que lo ejecutan
(`00_setup_config`, `01_ingesta_memoria_documental`,
`02_rag_chroma_retriever`, `Corrida_03_a_08`) — no un diseño aspiracional.

## 2. Problema de investigación

Redactar un estado del arte exige leer, comparar y sintetizar decenas de
artículos, verificar que cada afirmación tenga respaldo textual explícito, y
mantener trazabilidad completa entre cada oración generada y la evidencia
que la sostiene. Hacerlo manualmente es lento y propenso a errores de cita,
extrapolación o alucinación factual. Los LLM pueden generar texto fluido con
rapidez, pero sin control adicional no garantizan que cada afirmación esté
efectivamente respaldada por el corpus.

## 3. Dos capas operativas, no nueve etapas equivalentes

El sistema **no** es una cadena única de nueve etapas ejecutadas por el
mismo mecanismo. Son dos capas distintas, con mecanismos de ejecución
distintos, que cooperan mediante artefactos en disco:

**Capa A — preparación del experimento** (tres notebooks, ejecutados
manualmente, sin protocolo transaccional ni fingerprints):

```text
00_setup_config.ipynb
01_ingesta_memoria_documental.ipynb
02_rag_chroma_retriever.ipynb
```

Configura el experimento, ingesta y valida los documentos, aísla el
Ground Truth, y construye el índice de recuperación (Chroma) que usará
la capa siguiente. Requiere decisiones humanas (qué corpus usar, qué
tema, subir PDFs) y no tiene reintentos automáticos: cada notebook se
corre una vez, en orden, y valida sus propias precondiciones al
arrancar.

**Capa B — pipeline científico** (un notebook que invoca al orquestador
real, con estado persistente, fingerprints y reintentos):

```text
Corrida_03_a_08.ipynb
  └── src.orchestration.pipeline_orchestrator
        └── 03 → 03B → 04 → 05 → 06 ↔ 07 → 08
```

`Corrida_03_a_08.ipynb` no ejecuta la lógica científica por sí mismo —
invoca `src.orchestration.pipeline_orchestrator`, que administra el
estado transaccional (`ADVANCE`/`RETRY`/`RETURN`/`HALT_STAGE`/
`STOP_PIPELINE`) de las etapas 03 a 08. El ciclo `06 ↔ 07` (sección 12)
es parte de esa misma capa, no un paso adicional fuera de ella.

Las secciones 4 y 5 describen la Capa A con precisión; la sección 6
muestra exactamente qué artefactos conectan una capa con la otra; la
sección 10 describe la Capa B (`Corrida_03_a_08.ipynb`) en detalle.

## 4. Preparación del experimento: notebooks 00–02

**Importante**: 00, 01 y 02 no son tres `StageSpec` idénticos a las etapas
03–08. Son la capa de preparación documental/configuración, ejecutada
manualmente notebook por notebook, que produce los insumos que el pipeline
transaccional (03–08) consume después. No comparten el protocolo
PREPARE/EXECUTE/COMMIT de la sección 10, ni fingerprints, ni reintentos
automáticos — su idempotencia depende de que el usuario los ejecute en
orden y verifique las validaciones que cada uno imprime.

### 00 — Setup y configuración (`00_setup_config.ipynb`)

- Crea un experimento nuevo o retoma/sobrescribe uno existente
  (`experimento_paper_NN/`), preguntando interactivamente o con valores por
  defecto.
- Mantiene `active_experiment.json`, en la raíz de `PROJECT_DIR`, como el
  **selector** del experimento activo — todo lo que viene después (01, 02,
  y el pipeline 03–08) lo lee para saber sobre qué experimento operar.
- Configura el perfil del experimento: tema, alcance, términos de dominio,
  longitud del estado del arte, idioma, modo de escritura (descriptivo/
  crítico), enfoque (métodos/resultados/balanceado) y estilo de citas.
- Centraliza las políticas metodológicas de las etapas siguientes
  (`EXTRACTION_POLICY`, `QUANTITATIVE_EXTRACTION_POLICY`,
  `THEMATIC_ANALYSIS_POLICY`, `INGESTION_POLICY`,
  `OUTLINE_GENERATION_POLICY`, `DRAFT_GENERATION_POLICY`,
  `VERIFICATION_POLICY`, `EVALUATION_POLICY`) y la política de RAG
  (incluido el aislamiento de Ground Truth, ver sección 5).
- Crea/configura la estructura de directorios del experimento
  (`00_ground_truth/`, PDFs de entrada, textos extraídos, chunks, Chroma,
  `05_outputs/`).
- Genera, vía `%%writefile`, los módulos runtime planos que **01 y 02**
  importan directamente: `src/config.py`, `src/generation_config.py`,
  `src/rag_policy.py`, `src/experiment_config.py`, `src/prompts.py`,
  `src/io_utils.py`, `src/pdf_utils.py`, `src/llm_utils.py`,
  `src/rag_utils.py`. Ver sección 9 sobre por qué esta superficie plana
  coexiste con el paquete `src/config/` sin conflicto real.
- Configura la credencial de OpenAI para todo el pipeline
  (`llm_utils.ensure_openai_key`): variable de entorno →
  `.runtime_secrets/openai_api_key.txt` local → solicitud interactiva
  única si ninguna de las anteriores existe. Queda disponible como
  variable de entorno para 01, 02 y para el pipeline 03–08 (que la lee vía
  `src/io/credentials.py`).
- Valida al final, recargando los módulos recién escritos, que las
  políticas necesarias estén completas (claves requeridas presentes en
  cada policy) antes de dar el setup por terminado.

### 01 — Ingesta y memoria documental (`01_ingesta_memoria_documental.ipynb`)

- Carga `src/config.py` (el runtime generado por 00) por ruta explícita
  (`importlib.util.spec_from_file_location`), precisamente para no
  depender de la resolución de import normal de Python entre ese archivo y
  el paquete `src/config/` (ver sección 9).
- Identifica el experimento activo y verifica sus entradas: exactamente un
  PDF de Ground Truth y los PDFs de referencia (permite subir los que
  falten si el entorno lo soporta).
- **Ground Truth físicamente separado del corpus de referencia**: viven en
  directorios distintos (`00_ground_truth/` vs. el directorio de PDFs de
  entrada) desde el primer momento.
- Extrae el texto de los papers de referencia (PyMuPDF) y del Ground
  Truth.
- Del Ground Truth extrae **únicamente** la sección real de Literature
  Review / Related Work (detección de encabezados de sección, filtrando
  falsos positivos como números de página o pies de página) — nunca el
  documento completo — junto con sus indicadores de citación (numéricos y
  autor-año).
- Divide los papers de referencia en chunks con identificadores únicos
  (`chunk_id`, `source_filename`).
- Detecta secciones de revisión (`is_review_section_chunk`) y bibliografía/
  contenido no sustantivo (`is_bibliography_chunk`) dentro de los propios
  papers de referencia, y aplica las políticas de exclusión de RAG
  (`excluded_from_rag`).
- Genera `chunks_clean_for_rag.csv`/`.jsonl` — el archivo limpio que 02
  indexará — junto con archivos de auditoría de lo excluido
  (`chunks_flagged_review_sections.csv`, `chunks_flagged_bibliography.csv`,
  `chunks_flagged_for_rag_policy.csv`,
  `chunks_flagged_non_substantive.csv`) y manifiestos
  (`pdf_validation_manifest.csv`, `references_text_manifest.csv`).

**El Ground Truth nunca entra a `chunks_clean_for_rag` y nunca se usa como
evidencia de generación** — sus artefactos (`ground_truth_full_text.txt`,
`ground_truth_literature_review.txt`, metadatos y citas) quedan en
`00_ground_truth/`, un directorio distinto al que 02 indexa.

### 02 — Construcción de RAG con ChromaDB (`02_rag_chroma_retriever.ipynb`)

- Carga la configuración del experimento activo producida por 00 (incluida
  la credencial de OpenAI ya configurada).
- Carga **únicamente** `chunks_clean_for_rag.csv` — nunca ningún archivo
  del Ground Truth.
- **Valida explícitamente que no haya Ground Truth dentro del archivo**
  antes de indexar: revisa que ningún chunk tenga
  `is_review_section_chunk`/`is_bibliography_chunk`/`excluded_from_rag` en
  `True`, ni texto vacío, ni un `source_filename` que sugiera Ground Truth
  — si encuentra cualquiera de esas condiciones, se detiene con error en
  vez de indexar.
- Reconstruye la colección Chroma oficial del experimento: elimina
  primero cualquier colección con nombres prohibidos/deprecados
  (`ground_truth_refs` está explícitamente bloqueado — el código rechaza
  incluso *nombrar* la colección oficial así) y la colección anterior del
  mismo experimento, antes de recrearla desde cero.
- Calcula embeddings (`sentence-transformers`) e indexa los chunks
  permitidos, conservando `source_filename` y `chunk_id` como metadatos de
  cada entrada — la trazabilidad que toda cita posterior del sistema
  reutiliza.
- Genera `chroma_index_manifest.json`.
- Expone el retriever (con diversidad y trazabilidad) que las etapas
  posteriores usan transversalmente, y corre pruebas/auditorías de
  recuperación sobre el propio notebook antes de darlo por válido.

**Principio verificado en el código, no solo documentado**:

```text
PDFs de referencia → extracción → chunks → filtrado → chunks_clean_for_rag → Chroma
```

**Nunca**:

```text
Ground Truth → Chroma
```

## 5. Ground Truth: aislamiento verificado en tres puntos independientes

1. **01**: Ground Truth y PDFs de referencia viven en directorios
   distintos desde la ingesta; del Ground Truth solo se extrae su sección
   de Literature Review/Related Work, nunca se convierte en chunks para
   RAG.
2. **02**: antes de indexar, valida activamente que `chunks_clean_for_rag`
   no contenga ningún indicio de Ground Truth (por las columnas de
   exclusión y por el propio nombre de archivo), y bloquea explícitamente
   el nombre de colección `ground_truth_refs`.
3. **Pipeline 03–08**: Ground Truth se resuelve **exclusivamente** en la
   etapa 08 (`src/tools/evaluation/ground_truth.py`), para comparar el
   resultado final — ninguna etapa anterior (03–07) lee ni recibe
   contenido de Ground Truth, reforzado con listas de rechazo explícitas
   en varios módulos de `src/tools/` además de la validación de política
   en `src/adapters/verification_orchestrator_runtime.py`.

## 6. Cómo se conectan los notebooks 00–02 con el pipeline 03–08

```text
00 · Configuración del experimento
      │
      ├── active_experiment.json
      ├── políticas
      ├── parámetros del experimento
      ├── configuración runtime
      └── estructura de directorios
      │
      ▼
01 · Ingesta y memoria documental
      │
      ├── validación de PDFs
      ├── extracción de texto
      ├── Ground Truth separado
      ├── chunking
      ├── exclusiones para RAG
      └── chunks_clean_for_rag
      │
      ▼
02 · Chroma / RAG
      │
      ├── embeddings
      ├── índice vectorial
      ├── metadatos source_filename + chunk_id
      ├── manifest del índice
      └── retriever
      │
      ▼
Corrida_03_a_08.ipynb
      │
      ├── sincronización del código
      ├── pre-flight
      └── PipelineOrchestrator
              │
              ├── 03 Extracción científica
              ├── 03B Extracción cuantitativa
              ├── 04 Análisis temático
              ├── 05 Generación del esquema
              ├── 06 Redacción
              ├── 07 Verificación y trazabilidad
              └── 08 Evaluación
```

Este diagrama es la secuencia de **preparación → arranque**, no una
cadena de nueve `StageSpec` iguales: los tres primeros bloques son
Capa A (notebooks, sin estado transaccional); todo lo que cuelga de
`PipelineOrchestrator` es Capa B, administrada por
`src.orchestration.pipeline_orchestrator` con fingerprints, reintentos y
transiciones explícitas (sección 19).

**Qué artefacto concreto cruza cada frontera** (no son los notebooks en
sí los que se comunican, son estos archivos en disco):

- `00 → 01`: `active_experiment.json` (selector del experimento activo)
  + las políticas y la estructura de directorios que 00 deja escritos —
  01 falla explícitamente si `active_experiment.json` o los módulos
  runtime que 00 genera no existen.
- `01 → 02`: `chunks_clean_for_rag.csv`/`.jsonl` + sus manifiestos de
  auditoría — 02 rechaza continuar si ese archivo no existe o no pasa su
  validación de Ground Truth.
- `02 → Corrida_03_a_08.ipynb`: la colección Chroma persistente +
  `chroma_index_manifest.json` — la Capa B recupera evidencia contra esa
  colección ya construida, nunca reconstruye el índice por sí misma.
- `Corrida_03_a_08.ipynb → PipelineOrchestrator (03 en adelante)`: a
  partir de aquí ya no hay archivos que un notebook anterior "entregue"
  al siguiente — todo se coordina por artefactos dentro de
  `{experimento}/05_outputs/` y el estado transaccional en
  `pipeline_state.json` (secciones 11 y 19), propio del `StateStore`/
  fingerprints del orquestador.

## 7. Arquitectura del pipeline 03–08

### Agentes (lógica científica de cada etapa)

Viven en `src/agents/`. Cada agente encapsula la decisión científica de su
etapa (qué extraer, cómo redactar, cómo verificar un claim) y es agnóstico
de persistencia, red o LLM concretos — recibe sus dependencias inyectadas.

```text
src/agents/extraction_agent.py
src/agents/thematic_analysis_agent.py
src/agents/outline_generation_agent.py
src/agents/draft_writing_agent.py
src/agents/verification_agent.py
```

### Infraestructura (orquestación, estado, contratos)

No contiene lógica científica. Sostiene el pipeline como sistema
transaccional:

```text
src/orchestration/   StageSpec, run_stage(), run_pipeline(), decision_engine
src/state/           StateStore, PipelineState, fingerprints
src/contracts/       AgentInput, AgentResult (contrato uniforme de I/O de cada agente)
src/bootstrap/       preparación del proyecto y del experimento
src/io/              escritura atómica, credenciales
src/adapters/        conecta agentes + runtime real (LLM, Chroma, disco) con el contrato de StageSpec
src/runtime/         protocolos de ejecución (build_execution/runtime_transaction/resolve_resume) por etapa
src/config/          políticas y parámetros por etapa, sin valores por defecto silenciosos
```

### Módulos de evaluación (etapa 08, sin agente propio)

`src/tools/evaluation/` no define un "agente" en el sentido de
`src/agents/` — es un conjunto de módulos puros (normalización, ROUGE-L,
similitud semántica, BERTScore, auditoría numérica, auditoría de claims y
citas, LLM Judge) ensamblados por `src/tools/evaluation/evaluation_pipeline.py`
y conectados al contrato transaccional mediante
`src/adapters/evaluation_orchestrator_runtime.py`. La diferencia con los
agentes 03-07 es deliberada: 08 no toma una decisión científica única y
reintentable como los agentes — compone y persiste métricas.

## 8. El pipeline científico (etapas 03–08) — qué hace cada una

| Etapa | Nombre | Qué hace |
|---|---|---|
| 03 | Extracción de KB y Corpus Eligibility | Construye fichas por paper (problema, métodos, datasets, resultados), clasifica cada documento en `INCLUDE`/`EXCLUDE`/`QUARANTINE` (ver sección 8.1) y consolida las fichas `INCLUDE` en la base de conocimiento estructurada. |
| 03B | Extracción cuantitativa | Identifica y normaliza métricas y resultados numéricos, vinculados a paper y chunk. |
| 04 | Análisis temático | Agrupa métodos/datasets/resultados en temas y comparaciones. |
| 05 | Generación del esquema | Convierte el análisis temático en la estructura de secciones del estado del arte. |
| 06 | Redacción del borrador | Redacta cada sección citando `[fuente.pdf \| chunk_id]`, con RAG real contra la colección construida en 02. |
| 07 | Verificación factual y trazabilidad | Descompone el borrador en claims, verifica cada uno contra el corpus, clasifica veredicto y elegibilidad de corrección. |
| 08 | Evaluación experimental | Compara contra Ground Truth con métricas automáticas, LLM Judge y métricas factuales; persiste los 15 outputs finales. |

### 8.1. Corpus Eligibility Gate (etapa 03) — dos fases

Un documento individual no útil o no validable nunca detiene el corpus
completo. `src/tools/extraction/corpus_eligibility.py` clasifica cada
ficha en dos fases sucesivas, ambas antes de que el quality gate
científico (campos como `target_domain`/`methods_or_models`/`main_results`)
se aplique:

- **Fase 1 — pre-elegibilidad** (justo después del repair de título,
  antes de la clasificación de relevancia): review ya confirmado →
  `EXCLUDE`; título irrecuperable o evidencia insuficiente → `QUARANTINE`;
  el resto → `CANDIDATE`, continúa a fase 2.
- **Fase 2 — elegibilidad final** (después de la clasificación de
  relevancia): fuera de scope/dominio excluido → `EXCLUDE`; relevancia
  indeterminable → `QUARANTINE`; el resto → `INCLUDE`.

Solo `INCLUDE` entra al plan de revisión científico y puede requerir
retry por campos faltantes. `EXCLUDE`/`QUARANTINE` nunca aparecen ahí, y
quedan auditados en `scientific_cards_quarantine_audit.csv` /
`scientific_cards_review_exclusion_audit.csv`. Stage03 solo hace `HALT`
global por una condición sistémica — el corpus elegible (`INCLUDE`) cae
por debajo de `extraction_policy.corpus_eligibility_policy.min_include_
corpus_size` (default 1) — nunca por un documento individual.

`extraction_policy.exclude_reviews` es `True` por defecto para todo
experimento nuevo (política metodológica: reviews/surveys completos no
forman parte del corpus de generación) — sigue siendo configurable
explícitamente. `target_domain` solo se exige como campo crítico para
papers que reclaman un dominio de aplicación concreto — un paper
metodológico/fundacional no se bloquea por carecer de un dominio que su
contribución no reclama, y nunca se le inventa uno.

## 9. Superficies de configuración: `src/config.py` y `src/config/`

`src/config.py` (archivo plano) y `src/config/` (paquete, con
`generation_policy_config.py` y las policies de cada etapa) coexisten
deliberadamente, con consumidores reales y distintos:

- `src/config.py` (y los módulos planos hermanos listados en la sección
  4) los escribe **00** y los consumen **01 y 02** — la capa de
  preparación documental, ejecutada por notebook, fuera del orquestador.
  `01` incluso carga `src/config.py` por ruta explícita
  (`importlib.util.spec_from_file_location`) precisamente para no
  depender de cómo Python resolvería el import si solo escribiera
  `import config` — una decisión de diseño explícita en el propio
  código, no un descuido.
- `src/config/` (el paquete) lo consume el pipeline transaccional
  (`src/orchestration/`, `src/adapters/`, `src/agents/`, `src/tools/`) —
  valida y aplica defaults canónicos (ver sección 8.1) que la capa plana
  no necesita replicar.

Ambas superficies aplican la misma política de aislamiento de Ground
Truth cada una en su propio código — no es una duplicación accidental
sin resolver, sino dos capas con responsabilidades y consumidores
distintos que hoy coinciden en esa política. La fuente de verdad para
todo lo que ejecuta el orquestador (03→08 vía `StateStore`/`StageSpec`)
es siempre `active_experiment.json`, nunca los módulos planos.

## 10. `Corrida_03_a_08.ipynb` — cómo se ejecuta el pipeline científico en la práctica

Es la forma operativa real con la que se ejecuta el sistema hoy. Hace, en
orden:

### 1. Sincroniza el código

Clona una copia fresca del repositorio de GitHub en un directorio aparte,
y hace *overlay* (`shutil.copytree`/`copy2`) sobre `/content/proyecto_
estado_arte`, excluyendo explícitamente `.git`, `active_experiment.json`,
`venv_estado_arte`, y cualquier carpeta `experimento_paper_*`. Por
construcción, actualizar el código con esta celda **nunca** borra
experimentos existentes ni cambia el experimento activo seleccionado.
Cualquier edición local no confirmada en GitHub, en cambio, se pierde en
este paso — el código real que se ejecuta siempre es el que está en el
repositorio remoto, no el que pudiera haberse editado directamente en la
sesión de Colab.

### 2. Verifica/prepara el entorno

Crea el entorno virtual en `/content/venv_estado_arte` si no existe,
sincroniza `requirements.txt` (`pip install` + `pip check`), y confirma
en caliente que el Python del venv puede importar las dependencias
básicas. Todo lo que sigue usa el Python de ese venv, nunca el global.

### 3. Pre-flight del experimento

Antes de tocar 03–08: lee `active_experiment.json`, resuelve el
directorio del experimento activo, y lista los PDFs de referencia
recorriendo el directorio del experimento **excluyendo siempre** el
subdirectorio de Ground Truth. También fija explícitamente dos políticas
operativas de esta corrida concreta antes de arrancar:
`evaluation_policy.allow_partial_halt_evaluation = True` (ver punto 5) y
`draft_generation_policy.draft_representation_contract =
"canonical_sentences_v2"`, con verificación fail-closed de que quedó
persistido.

### 4. Ejecuta el pipeline científico mediante el orquestador

El notebook invoca el orquestador real:

```bash
python -m src.orchestration.pipeline_orchestrator \
  --project-dir /content/proyecto_estado_arte \
  --start-stage 03_agente_extraccion_kb \
  --until 08_evaluacion_experimental
```

El **orquestador**, no el notebook, administra estado persistente,
fingerprints, freshness, reintentos, transiciones
(`ADVANCE`/`RETRY`/`RETURN`/`HALT_STAGE`/`STOP_PIPELINE`), reutilización
de artefactos vigentes, e invalidación de etapas dependientes. El
notebook solo lanza el comando y lee su código de salida — si es
distinto de cero, se detiene con un error explícito, sin recomendar
`--force-rerun` a ciegas.

### 5. `PARTIAL_HALT` en verificación — nunca implica aprobación para publicación

Si 07 emite un `HALT_STAGE` por agotamiento científico del ciclo
06↔07 (ver sección 12) mientras el resultado sigue siendo válido como
objeto experimental, el notebook comprueba si `agent07_runtime_report.
json` existe y si `allow_partial_halt_evaluation` está en `True` (lo
fija así en el paso 3, para esta corrida). Si ambas condiciones se
cumplen, **solicita** al orquestador que ejecute la etapa 08 por
separado:

```bash
python -m src.orchestration.pipeline_orchestrator \
  --project-dir /content/proyecto_estado_arte \
  --start-stage 08_evaluacion_experimental \
  --until 08_evaluacion_experimental
```

El propio orquestador decide si realmente la ejecuta o la reconoce como
`SKIPPED_FRESH` — el notebook nunca asume que un `evaluation_summary.
json` previo sigue vigente. Esto **no es un bypass arbitrario** del
verificador: la política del pipeline (el flag explícito en
`evaluation_policy`) es la que determina si la evaluación está permitida
tras un halt científico, y el `decision_log` de 07 nunca se reescribe —
su entrada sigue diciendo `HALT_STAGE` igual que antes.

**Distinción explícita que el propio manifest de evaluación persiste**
(`evaluation_manifest.json["pipeline_outcome"]`): `approved_for_
publication` y `usable_for_evaluation` son campos separados. Un
resultado puede no estar listo para publicación (`approved_for_
publication=false`) pero seguir siendo un objeto experimental válido
para la etapa 08 (`usable_for_evaluation=true`) — son dos preguntas
distintas, y el notebook las imprime por separado en su resumen final,
nunca las combina en una sola.

### 6. Resume el estado final

Lee `evaluation_summary.json` y `evaluation_manifest.json` y muestra:
resultado global, palabras generadas, palabras del Ground Truth,
métricas automáticas, puntajes del LLM Judge, métricas factuales, y el
`pipeline_outcome` completo (incluidos `verification_approved`,
`human_review_required`, `approved_for_publication`,
`usable_for_evaluation`, `agent07_halt_reason`).

### 7. Consolida resultados para tesis

La última celda **no vuelve a generar ciencia ni cambia ningún
resultado** — solo reorganiza y copia artefactos ya producidos por 03–08
hacia `{experimento}/05_outputs/08_final_results/`:

```text
estado_del_arte_generado.md / .json
resumen_experimento.csv
metricas_evaluacion.csv
trazabilidad_claims_final.csv
tabla_comparativa_papers.csv
resumen_por_seccion.csv
gaps_y_limitaciones.csv
resumen_resultado_final.xlsx
```

y empaqueta esa carpeta en un `.zip` descargable. Es consolidación y
presentación para la tesis, no una re-ejecución del pipeline.

## 11. Outputs principales

### Etapa 01 (ingesta)

- `chunks_clean_for_rag.csv`/`.jsonl` — la entrada oficial de 02.
- `chunks_flagged_review_sections.csv`, `chunks_flagged_bibliography.csv`,
  `chunks_flagged_for_rag_policy.csv`,
  `chunks_flagged_non_substantive.csv` — auditoría de lo excluido.
- `pdf_validation_manifest.csv`, `references_text_manifest.csv`.
- En `00_ground_truth/`: `ground_truth_full_text.txt`,
  `ground_truth_literature_review.txt`,
  `ground_truth_literature_review_metadata.json`,
  `ground_truth_literature_review_citation_ids.json`,
  `ground_truth_literature_review_author_year_citations.json`.

### Etapa 02 (RAG)

- Colección Chroma persistente del experimento.
- `chroma_index_manifest.json`.

### Etapa 03 (extracción y corpus eligibility)

- `scientific_cards.jsonl` — todas las fichas, con `corpus_eligibility`
  (`INCLUDE`/`EXCLUDE`/`QUARANTINE`) persistido para cada una.
- `scientific_cards_revision_plan.csv` — solo fichas `INCLUDE` inválidas.
- `scientific_cards_review_exclusion_audit.csv`,
  `scientific_cards_quarantine_audit.csv`.
- `scientific_cards_quality_check.csv`, `extraction_retrieval_trace.csv`,
  `scientific_knowledge_base.{csv,jsonl}`, `scientific_extraction_manifest.json`.

### Etapa 06 (borrador)

- `state_of_art_draft.json` — borrador con secciones, citas y claims.
- `draft_generation_manifest.json`, `draft_validation_report.json`,
  `draft_length_check.csv`, `draft_quality_check.csv`.

### Etapa 07 (verificación)

Cuatro artefactos científicos incondicionales, más uno condicional, en
`{experimento}/05_outputs/06_verification_traceability/`:

```text
provisional_verification_traceability_bundle.json
multi_proposal_resolution_result.json
agent07_runtime_report.json
agent07_artifact_manifest.json
writer_revision_request.json      (SOLO si la transición es RETURN)
```

### Etapa 08 — los 15 outputs obligatorios de evaluación

En `{experimento}/05_outputs/07_evaluation/`:

```text
 1. automatic_metrics.csv
 2. semantic_chunk_alignment.csv
 3. bertscore_chunk_alignment.csv
 4. factual_metrics.csv
 5. final_citation_check.csv
 6. final_claim_audit.csv
 7. llm_judge_evaluation.json
 8. llm_judge_scores.csv
 9. corpus_gap_suggestions.csv
10. corpus_gap_suggestions.md
11. final_selected_metrics.csv
12. evaluation_summary.json
13. final_evaluation_report.md
14. evaluation_validation_report.json
15. evaluation_manifest.json
```

`agent08_upstream_numeric_check.csv` **no** es uno de los 15 — es un
artefacto intermedio que 08 sobrescribe internamente.

### Consolidación final (`Corrida_03_a_08.ipynb`, paso 7)

En `{experimento}/05_outputs/08_final_results/`, ver sección 10.7.

## 12. El ciclo correctivo 06 ↔ 07

07 no es un simple validador de paso único. Cuando encuentra un claim
`PARTIALLY_SUPPORTED` con evidencia real disponible, construye una
propuesta de corrección localizada y, si la acepta, emite una transición
**`RETURN`** hacia 06 en vez de `ADVANCE`:

```text
06 redacta
   ↓
07 verifica -> claim corregible con evidencia real
   ↓
RETURN a 06, con una "writer_revision_request" trazable
   ↓
06 entra en modo REVISION: corrige SOLO la sección/claim señalado,
   el resto del borrador queda intacto
   ↓
07 reverifica el borrador corregido
   ↓
si el claim ya es SUPPORTED -> ADVANCE hacia 08
si el ciclo agota sus rondas sin aprobación completa -> HALT_STAGE (ver sección 10.5)
```

Este ciclo se persiste en disco de forma transaccional en
`writer_verifier_cycle/round_NN/` — 07 crea la ronda en
`AWAITING_REVISION`, 06 la completa (nunca la crea) a `REVISION_COMPLETED`,
y un segundo intento de completar la misma ronda se rechaza explícitamente.

### Identidad estable de claims (`claim_uid`)

`claim_id` (ej. `S5_C2`) es **posicional**: 06 lo recalcula cada vez que
regenera una sección completa en modo `REVISION`. Para rastrear un claim
de forma confiable entre rondas, cada claim tiene además `claim_uid`
(UUID4 opaco, minteado una sola vez), `claim_version` (entero monótono) y
`parent_claim_uids` (linaje explícito). La identidad se declara
explícitamente por 06 (`identity_action` = `CONTINUE`/`NEW`/
`SPLIT_CHILD`/`MERGE`) — nunca se infiere después por similitud de texto.
Ver `src/tools/draft_writing/claim_identity.py`.

## 13. RAG, memoria documental, KB científica y trazabilidad

- **RAG**: 06 recupera evidencia por sección al redactar contra la
  colección Chroma construida en 02; 07 puede además hacer una
  recuperación **independiente** por claim, etiquetando esa evidencia con
  `usage_role="SUPPORT"` (ver `src/adapters/verification_incremental_
  retriever.py`).
- **KB científica (etapas 03/03B/04)**: fichas por paper (solo las
  `INCLUDE`, ver sección 8.1), extracción cuantitativa y análisis
  temático, consolidadas antes de generar el esquema.
- **Trazabilidad**: toda afirmación del borrador lleva citas internas
  `[archivo.pdf | chunk_id]` — la misma unidad atómica que 01 asignó y 02
  indexó; 07 construye una matriz de trazabilidad completa que 08 audita.

## 14. Estructura real de carpetas

```text
tesis-sistema-multiagente-main/
├── README.md                    (este archivo)
├── LEEME_PRIMERO.md              (guía corta operativa)
├── requirements.txt
├── smoke_test.py                 (contrato transaccional genérico, con dobles)
├── smoke_test_draft.py           (fallo de build_execution -> FAILED, con dobles)
├── src/
│   ├── adapters/                 conecta agentes/runtime real con StageSpec
│   ├── agents/                   lógica científica de cada agente (03-07)
│   ├── bootstrap/                preparación de proyecto/experimento
│   ├── capabilities/              capacidades reutilizables (extracción cuantitativa)
│   ├── config/                   políticas por etapa (paquete, ver sección 9)
│   ├── config.py                 runtime plano generado por 00 (ver sección 9)
│   ├── contracts/                AgentInput / AgentResult
│   ├── io/                       escritura atómica, credenciales
│   ├── orchestration/            StageSpec, run_stage, run_pipeline, decision_engine
│   ├── runtime/                  protocolos build_execution/runtime_transaction/resolve_resume
│   ├── state/                    StateStore, PipelineState, fingerprints
│   └── tools/
│       ├── draft_writing/        herramientas de la etapa 06
│       ├── evaluation/           módulos puros de la etapa 08 (ROUGE-L, BERTScore, LLM Judge, ...)
│       ├── extraction/           herramientas de la etapa 03 (incluye corpus_eligibility.py, review_exclusion.py)
│       ├── outline_generation/   herramientas de la etapa 05
│       ├── quantitative_extraction/  herramientas de la etapa 03B
│       ├── thematic_analysis/    herramientas de la etapa 04
│       └── verification/         herramientas de la etapa 07
└── tests/
    ├── orchestration/            suites principales (auto-contenidas, ver sección 18)
    ├── evaluation/
    ├── verification/
    ├── fixtures/
    ├── integration/
    ├── v16/
    └── v17/
```

## 15. Requisitos

- Python **3.11** (versión exacta validada en corridas reales de Colab;
  el código en sí solo requiere `>= 3.10`).
- Entorno **aislado** vía `virtualenv` — nunca el Python global de Colab.
  `requirements.txt`/`constraints-colab.txt` fijan `chromadb==1.5.9`.
- Versiones exactas confirmadas: `langchain-core==0.3.86`,
  `langchain-openai==0.3.35`, `openai==1.109.1`, `chromadb==1.5.9`,
  `transformers==4.46.3`, `tokenizers==0.20.3`,
  `huggingface-hub==0.36.2`, `sentence-transformers==5.7.0`,
  `bert-score==0.3.13`, `numpy==2.4.6`, `pandas==3.0.5`,
  `scikit-learn==1.9.0`, `matplotlib==3.11.1`, `PyMuPDF==1.28.2`,
  `rouge-score==0.1.2`, `tabulate==0.10.0`, `cryptography==46.0.7`,
  `torch==2.13.0`.

## 16. Instalación y ejecución operativa (Colab)

```text
1. Ejecutar 00_setup_config.ipynb          -> crea/selecciona el experimento
2. Ejecutar 01_ingesta_memoria_documental.ipynb  -> chunks_clean_for_rag
3. Ejecutar 02_rag_chroma_retriever.ipynb  -> índice Chroma listo
4. Ejecutar Corrida_03_a_08.ipynb          -> sincroniza código, corre 03-08, consolida
```

Instalación mínima sin venv, solo para desarrollo/tests fuera de Colab:

```bash
python3 -m pip install -r requirements.txt
```

## 17. Formato esperado de `active_experiment.json`

Debe existir en la raíz de `PROJECT_DIR` (fuera de la carpeta del
experimento):

```json
{
  "active_experiment_id": "experimento_paper_02",
  "experiment_dir": "/content/proyecto_estado_arte/experimento_paper_02",
  "evaluation_policy": { "...": "..." }
}
```

- `active_experiment_id`/`experiment_dir` son obligatorios — sin ellos,
  ni el pre-flight de `Corrida_03_a_08` ni `run_pipeline()` pueden
  resolver el experimento.
- `evaluation_policy` es **obligatorio para llegar a la etapa 08** — debe
  ser un diccionario no vacío. No tiene valores por defecto silenciosos.
- `extraction_policy` es opcional — si se omite (o se omiten campos como
  `exclude_reviews`/`corpus_eligibility_policy`), la política
  metodológica canónica se aplica por defecto (ver sección 8.1).

## 18. Cómo ejecutar pruebas

**No uses `pytest` para `tests/orchestration/`** (ni para varios archivos
de `tests/v16/`). Verificado empíricamente: esas suites usan un decorador
`@scenario` que captura toda excepción internamente sin relanzarla — bajo
`pytest`, cada función `test_*` se marca `passed` aunque la aserción
interna haya fallado realmente. El corredor confiable es ejecutar cada
archivo directamente:

```bash
for f in tests/orchestration/test_*.py; do python3 "$f" || echo "FALLÓ: $f"; done
```

Cada archivo imprime `N/N escenarios OK` y termina con código de salida 0
si todo pasó.

## 19. PREPARE / EXECUTE / COMMIT / RESUME (etapas 03-08)

Cada etapa se ejecuta con el mismo protocolo transaccional
(`src/state/state_store.py`, `run_stage()` en
`src/orchestration/pipeline_orchestrator.py`):

1. **PREPARE**: `store.prepare_execution(...)` registra la intención de
   ejecutar. Si ya hay una ejecución pendiente sin comprometer, lanza
   `RuntimeError`.
2. **EXECUTE**: se corre `build_execution` + la lógica real del agente.
3. **Persistencia del resultado**: `store.persist_agent_result(...)`
   guarda el `AgentResult` en disco antes de comprometerlo.
4. **COMMIT**: `store.commit_execution(...)` es la única escritura que
   marca la etapa como `COMPLETED`/`FAILED`, junto con sus fingerprints.
5. **RESUME**: si se reintenta una etapa con una ejecución pendiente sin
   comprometer, `resolve_resume` decide si el resultado ya persistido
   puede comprometerse directamente o si hace falta reejecutar.

## 20. Fingerprints y reconstrucción

`src/state/fingerprints.py` calcula un fingerprint compuesto por etapa a
partir de tres partes: `input_fingerprint`, `config_fingerprint`,
`dependencies_fingerprint`. Si cualquiera cambia, `run_stage()` decide
reconstruir esa etapa en vez de reutilizar el resultado `COMPLETED`
anterior (`SKIPPED_FRESH` solo ocurre cuando el fingerprint compuesto
coincide exactamente).

## 21. Reproducibilidad

- Las funciones puras de `src/tools/` no leen ni escriben archivos ni
  llaman a red — reciben sus datos de entrada y devuelven estructuras en
  memoria.
- `langdetect` fija `DetectorFactory.seed = 0` para detección de idioma
  determinista.
- Los fingerprints (sección 20) permiten verificar si una corrida
  reproduce exactamente las mismas entradas/configuración/dependencias.
- El componente **no determinista real** es el propio LLM (OpenAI): el
  pipeline no fija `temperature=0` de forma global; la reproducibilidad
  exacta del texto generado no está garantizada entre corridas.

## 22. Estado de validación

- Ya existen corridas científicas reales del pipeline completo con
  OpenAI y Chroma reales (no solo dobles deterministas), incluida al
  menos una corrida real de regresión ejecutada después de una serie de
  cambios estructurales sobre el código de las etapas 03/06/07 — sin
  diferencias de comportamiento observadas en los casos verificados
  (`INCLUDE`/`EXCLUDE`/`QUARANTINE`, `quality_status`,
  `requested_transition`, `failure_reason_codes`).
- El ciclo cualitativo completo `06 → 07 (RETURN) → 06 (REVISION) → 07
  (ADVANCE)` está probado de punta a punta con código productivo real.
- La etapa 08 completa (métricas automáticas, factuales, LLM Judge,
  persistencia de los 15 outputs, fingerprints, contrato transaccional)
  está migrada y probada.

## 23. Relación con la tesis

Este repositorio es el soporte de implementación de la tesis. El diseño
de dos partes cooperantes (preparación documental 00–02 + pipeline
transaccional 03–08), el ciclo correctivo 06↔07, y las métricas de
evaluación de la etapa 08 corresponden directamente a los objetivos
específicos y la metodología descritos en el documento de tesis.
