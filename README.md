# Sistema multiagente para generación y evaluación automatizada de estados del arte

## 1. Nombre y propósito del proyecto

Este repositorio contiene el código fuente reutilizable y las pruebas
automatizadas de un **sistema multiagente basado en modelos de lenguaje
grandes (LLM)** que genera, verifica y evalúa automáticamente **estados del
arte** (revisiones de literatura científica) a partir de un corpus de
artículos proporcionado por el usuario.

Es el soporte de código de una tesis de maestría. Este `README.md` describe
el **estado real y verificado del código** al momento de esta entrega — no
un diseño aspiracional.

## 2. Problema de investigación

Redactar un estado del arte exige leer, comparar y sintetizar decenas de
artículos, verificar que cada afirmación tenga respaldo textual explícito, y
mantener trazabilidad completa entre cada oración generada y la evidencia
que la sostiene. Hacerlo manualmente es lento y propenso a errores de cita,
extrapolación o alucinación factual. Los LLM pueden generar texto fluido con
rapidez, pero sin control adicional no garantizan que cada afirmación esté
efectivamente respaldada por el corpus.

## 3. Objetivo general del sistema

Diseñar e implementar un pipeline multiagente que:

1. procese un corpus de papers científicos y construya una base de
   conocimiento estructurada y trazable;
2. redacte un estado del arte por secciones, citando explícitamente la
   fuente y el fragmento (`chunk`) de cada afirmación;
3. verifique automáticamente cada afirmación contra el corpus, clasifique su
   nivel de soporte y, cuando sea posible, proponga una corrección
   localizada respaldada por evidencia real;
4. cierre un ciclo correctivo real entre el redactor y el verificador antes
   de dar por definitivo el borrador;
5. evalúe el resultado final con métricas automáticas, un LLM Judge y
   métricas factuales y de trazabilidad, comparándolo contra un Ground
   Truth cuando existe.

## 4. Arquitectura multiagente actual

El sistema distingue tres capas:

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
agentes 02-07 es deliberada: 08 no toma una decisión científica única y
reintentable como los agentes — compone y persiste métricas.

## 5. Flujo completo

```text
00 → 01 → 02 → 03 → 03B → 04 → 05 → 06 → 07 ↔ 06 → 07 → 08
```

| Etapa | Nombre | Qué hace |
|---|---|---|
| 00 | Orquestación y planificación | Prepara el experimento, resuelve configuración común, inicializa `pipeline_state.json`. |
| 01 | Ingesta y preparación documental | Organiza los PDF de entrada, produce chunks limpios para RAG. |
| 02 | Extracción de información científica | Construye fichas por paper (problema, métodos, datasets, resultados, limitaciones). |
| 03 | Extracción de KB | Consolida las fichas en una base de conocimiento estructurada. |
| 03B | Extracción cuantitativa | Identifica y normaliza métricas y resultados numéricos, vinculados a paper y chunk. |
| 04 | Análisis temático | Agrupa métodos/datasets/resultados en temas y comparaciones. |
| 05 | Generación del esquema | Convierte el análisis temático en la estructura de secciones del estado del arte. |
| 06 | Redacción del borrador | Redacta cada sección citando `[fuente.pdf \| chunk_id]`, con RAG real. |
| 07 | Verificación factual y trazabilidad | Descompone el borrador en claims, verifica cada uno contra el corpus, clasifica veredicto y elegibilidad de corrección. |
| 08 | Evaluación experimental | Compara contra Ground Truth con métricas automáticas, LLM Judge y métricas factuales; persiste los 15 outputs finales. |

## 6. El ciclo correctivo 06 ↔ 07

07 no es un simple validador de paso único. Cuando encuentra un claim
`PARTIALLY_SUPPORTED` con evidencia real disponible, construye una
propuesta de corrección localizada (`propose_correction`) y, si la acepta,
emite una transición **`RETURN`** hacia 06 en vez de `ADVANCE`:

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
```

Este ciclo se persiste en disco de forma transaccional en
`writer_verifier_cycle/round_NN/` (ver sección 17) — 07 crea la ronda en
`AWAITING_REVISION`, 06 la completa (nunca la crea) a `REVISION_COMPLETED`,
y un segundo intento de completar la misma ronda se rechaza explícitamente
(no hay doble escritura silenciosa).

### 07C: excluido del flujo activo

`07C` (reverificación de una corrección ya aplicada automáticamente) **no
forma parte del flujo activo**. La ruta real es `07 → 06 (RETURN) → 07`
directamente, sin pasar por 07C. El código conserva compatibilidad
histórica con 07C únicamente en `src/adapters/agent07c_handoff.py` y
mensajes/nombres de archivo heredados donde era inevitable — su presencia
en el código no significa que participe del registro de etapas activo
(`STAGE_ORDER` en `src/orchestration/pipeline_orchestrator.py` no lo
incluye).

## 7. RAG, memoria documental, KB científica y trazabilidad

- **Memoria documental (etapa 01)**: los PDF de entrada se convierten en
  chunks limpios, identificados por `(source_filename, chunk_id)` — la
  unidad atómica de cita en todo el pipeline.
- **RAG**: 06 recupera evidencia por sección al redactar; 07 puede además
  hacer una recuperación **independiente** por claim (no solo heredar la
  evidencia que 06 citó), etiquetando esa evidencia con
  `usage_role="SUPPORT"` — la única vía productiva real que asigna ese rol
  concreto (ver `src/adapters/verification_incremental_retriever.py`).
- **KB científica (etapas 02/03/03B/04)**: fichas por paper, extracción
  cuantitativa y análisis temático, consolidadas antes de generar el
  esquema.
- **Trazabilidad**: toda afirmación del borrador lleva citas internas
  `[archivo.pdf | chunk_id]`; 07 construye una matriz de trazabilidad
  completa (`claim_traceability_rows` + `claim_evidence_traceability_rows`
  + `correction_traceability_rows`) que 08 audita.

## 8. Estructura real de carpetas

```text
tesis-sistema-multiagente-main/
├── README.md                    (este archivo)
├── LEEME_PRIMERO.md              (guía corta operativa)
├── COLAB_SMOKE_TEST.md           (procedimiento de smoke test real)
├── requirements.txt
├── smoke_test.py                 (contrato transaccional genérico, con dobles)
├── smoke_test_draft.py           (fallo de build_execution -> FAILED, con dobles)
├── src/
│   ├── adapters/                 conecta agentes/runtime real con StageSpec
│   ├── agents/                   lógica científica de cada agente (02-07)
│   ├── bootstrap/                preparación de proyecto/experimento
│   ├── capabilities/              capacidades reutilizables (extracción cuantitativa)
│   ├── config/                   políticas por etapa
│   ├── contracts/                AgentInput / AgentResult
│   ├── io/                       escritura atómica, credenciales
│   ├── orchestration/            StageSpec, run_stage, run_pipeline, decision_engine
│   ├── runtime/                  protocolos build_execution/runtime_transaction/resolve_resume
│   ├── state/                    StateStore, PipelineState, fingerprints
│   └── tools/
│       ├── draft_writing/        herramientas de la etapa 06
│       ├── evaluation/           módulos puros de la etapa 08 (ROUGE-L, BERTScore, LLM Judge, ...)
│       ├── extraction/           herramientas de la etapa 02
│       ├── outline_generation/   herramientas de la etapa 05
│       ├── quantitative_extraction/  herramientas de la etapa 03B
│       ├── thematic_analysis/    herramientas de la etapa 04
│       └── verification/         herramientas de la etapa 07
└── tests/
    ├── orchestration/            suites principales (auto-contenidas, ver sección 13)
    ├── evaluation/
    ├── verification/
    ├── fixtures/
    ├── integration/
    ├── v16/
    └── v17/
```

**Nota sobre nombres de carpeta**: las carpetas de herramientas usan el
nombre completo de cada etapa (`outline_generation`, `quantitative_extraction`,
`thematic_analysis`), no abreviaturas — es el nombre real en disco, verificado
en esta entrega.

## 9. Requisitos

- Python **>= 3.10** (el código usa `str | None` y `list[str]` sin
  `from __future__ import annotations` en algunos módulos).
- Ver `requirements.txt` para las dependencias exactas, con una nota junto
  a cada una indicando si su versión fue verificada localmente (con dobles
  deterministas) o si requiere el smoke test real de Colab para
  confirmarse (LLM, Chroma).

## 10. Instalación

```bash
python3 -m pip install -r requirements.txt
```

Para el procedimiento completo de instalación + smoke test en Colab, ver
`COLAB_SMOKE_TEST.md`.

## 11. Configuración de `OPENAI_API_KEY`

`src/io/credentials.py` resuelve credenciales en este orden:

1. Variable de entorno `OPENAI_API_KEY`, si está definida.
2. Dentro de Google Colab, `google.colab.userdata` (el "Secrets" del
   notebook) — este camino se importa de forma diferida y solo se activa si
   el módulo `google.colab` existe.
3. Un archivo cifrado local (vía `cryptography.fernet.Fernet`), para uso
   fuera de Colab sin exponer la clave en texto plano.

Fuera de Colab, la vía más simple es:

```bash
export OPENAI_API_KEY="sk-..."
```

## 12. Formato esperado de `active_experiment.json`

Debe existir en la raíz de `PROJECT_DIR` (fuera de la carpeta del
experimento), con al menos:

```json
{
  "active_experiment_id": "exp_paper_02",
  "run_id": "run_2026_08_07",
  "evaluation_policy": { "...": "..." }
}
```

- `active_experiment_id` es **obligatorio** — sin él, `run_pipeline()`
  lanza `FileNotFoundError` indicando que falta correr la etapa 00.
- `run_id` es opcional; si falta, se usa el mismo valor que
  `active_experiment_id`.
- `evaluation_policy` es **obligatorio para llegar a la etapa 08** — debe
  ser un diccionario no vacío (`build_execution_for_stagespec` lo valida
  explícitamente y lanza `ValueError` si falta o está vacío). No tiene
  valores por defecto silenciosos.
- El directorio del experimento se resuelve como
  `PROJECT_DIR/{active_experiment_id}/`.

## 13. Cómo ejecutar pruebas

**No uses `pytest` para `tests/orchestration/`.** Verificado empíricamente:
esas suites usan un decorador `@scenario` que captura toda excepción
internamente y la registra en una lista `RESULTS` sin relanzarla — bajo
`pytest`, cada función `test_*` termina sin excepción y se marca `passed`
**aunque la aserción interna haya fallado realmente**. El corredor
confiable es ejecutar cada archivo directamente:

```bash
python3 tests/orchestration/test_writer_revision_cycle_core.py
python3 tests/orchestration/test_writer_verifier_cycle_e2e.py
python3 tests/orchestration/test_writer_verifier_round_conflict_integration.py
python3 tests/orchestration/test_agent07_manifest_conditional.py
python3 tests/orchestration/test_verification_claim_canonicalization.py
python3 tests/orchestration/test_verification_numeric_risk_characterization.py
python3 tests/orchestration/test_qualitative_correction_keyerror_and_return.py
```

Cada archivo imprime `N/N escenarios OK` y termina con código de salida 0
si todo pasó. Para correr **todas** las suites de `tests/orchestration/`:

```bash
for f in tests/orchestration/test_*.py; do python3 "$f" || echo "FALLÓ: $f"; done
```

Estado verificado en esta entrega: **468/468** en 37 suites.

## 14. Cómo ejecutar el pipeline

```bash
python3 -m src.orchestration.pipeline_orchestrator --project-dir /ruta/a/PROJECT_DIR
```

`run_pipeline()` no es un `for` fijo sobre las etapas: interpreta la
`RequestedTransition` real que devuelve cada etapa (`ADVANCE`, `RETRY`,
`RETURN`, `HALT_STAGE`, `STOP_PIPELINE`). Por eso el mismo comando cubre el
ciclo `06 ↔ 07` sin ningún flag especial: si 07 emite `RETURN`, el bucle
vuelve a 06 automáticamente; cuando 07 finalmente emite `ADVANCE`, sigue
hacia 08.

### Ejecutar hasta una etapa específica

```bash
python3 -m src.orchestration.pipeline_orchestrator --project-dir /ruta/a/PROJECT_DIR --until 07_agente_verificador
```

`--until` acepta cualquier clave de `STAGE_ORDER` y detiene el bucle apenas
esa etapa produce un resultado (incluso si pedía `ADVANCE`).

### Usar `--force-rerun`

```bash
python3 -m src.orchestration.pipeline_orchestrator --project-dir /ruta/a/PROJECT_DIR --force-rerun
```

Reejecuta la etapa inicial (`--until` o la primera) aunque ya esté
`COMPLETED` y vigente según fingerprints en `pipeline_state.json`. Las
etapas alcanzadas después por `ADVANCE`/`RETRY`/`RETURN` siguen evaluando
sus propios fingerprints normalmente — `--force-rerun` no las fuerza a
todas.

## 15. Outputs principales

### Etapa 06 (borrador)

- `state_of_art_draft.json` — borrador con secciones, citas y claims.
- `draft_generation_manifest.json`, `draft_validation_report.json`,
  `draft_length_check.csv`, `draft_quality_check.csv`.

### Etapa 07 (verificación)

Cuatro artefactos científicos incondicionales, más uno condicional, en
`PROJECT_DIR/{active_experiment_id}/05_outputs/06_verification_traceability/`
(nombre de carpeta heredado del notebook original, no de la clave del
`StageSpec`):

```text
provisional_verification_traceability_bundle.json
multi_proposal_resolution_result.json
agent07_runtime_report.json
agent07_artifact_manifest.json
writer_revision_request.json      (SOLO si la transición es RETURN)
```

### Etapa 08 — los 15 outputs obligatorios de evaluación

En `PROJECT_DIR/{active_experiment_id}/05_outputs/07_evaluation/` (mismo
motivo de numeración heredada):

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
artefacto intermedio que 08 sobrescribe internamente, no un output final
auditado (`src/adapters/evaluation_persistence.py`,
`INTERMEDIATE_NUMERIC_CHECK_FILENAME`).

## 16. Ubicación de `pipeline_state.json`

```text
PROJECT_DIR/{active_experiment_id}/05_outputs/00_orchestrator_planner/pipeline_state.json
```

Es el estado transaccional canónico de **todo** el pipeline (todas las
etapas comprometidas, sus fingerprints, su `AgentResult` persistido).

## 17. Ubicación de `writer_verifier_cycle/round_NN`

```text
PROJECT_DIR/{active_experiment_id}/05_outputs/writer_verifier_cycle/round_01/
```

Cada ronda del ciclo 06↔07 vive en su propia carpeta numerada, con al
menos:

```text
writer_revision_request.json     (de 07, al crear la ronda)
input_draft_reference.json       (de 07)
agent07_result.json              (de 07)
transition.json                  (de 07)
fingerprints.json                (de 07)
revised_draft.json               (de 06, al completar la ronda)
revision_changelog.json          (de 06)
revision_resolution_matrix.json  (de 06)
unresolved_issues.json           (de 06)
fingerprint.json                 (de 06)
_round_status.json               (estado interno: AWAITING_REVISION -> REVISION_COMPLETED)
```

## 18. PREPARE / EXECUTE / COMMIT / RESUME

Cada etapa se ejecuta con el mismo protocolo transaccional
(`src/state/state_store.py`, `run_stage()` en
`src/orchestration/pipeline_orchestrator.py`):

1. **PREPARE**: `store.prepare_execution(...)` registra la intención de
   ejecutar, generando un `decision_id`. Si ya hay una ejecución pendiente
   sin comprometer, lanza `RuntimeError` — nunca hay dos PREPARE
   simultáneos sin resolver.
2. **EXECUTE**: se corre `build_execution` + la lógica real del agente
   (fuera del `StateStore`).
3. **Persistencia del resultado**: `store.persist_agent_result(...)` guarda
   el `AgentResult` en disco antes de comprometerlo — permite recuperarlo
   si el proceso muere entre EXECUTE y COMMIT.
4. **COMMIT**: `store.commit_execution(...)` es la única escritura que
   marca la etapa como `COMPLETED`/`FAILED` en `pipeline_state.json`,
   junto con sus fingerprints.
5. **RESUME**: si se vuelve a llamar `run_stage()` sobre una etapa con una
   ejecución pendiente sin comprometer, `resolve_resume` decide si el
   resultado ya persistido puede comprometerse directamente (sin
   reinvocar al agente/LLM) o si hace falta reejecutar.

## 19. Fingerprints y reconstrucción

`src/state/fingerprints.py` calcula un fingerprint compuesto por etapa a
partir de tres partes independientes: `input_fingerprint`,
`config_fingerprint`, `dependencies_fingerprint` (`build_stage_fingerprints`).
Si cualquiera de las tres cambia, el fingerprint compuesto cambia, y
`run_stage()` decide reconstruir esa etapa en vez de reutilizar el
resultado `COMPLETED` anterior (`SKIPPED_FRESH` solo ocurre cuando el
fingerprint compuesto coincide exactamente). El fingerprint aprobado de 07
se propaga a 08 como `upstream_fingerprint` en la firma de evaluación
(`src/adapters/evaluation_fingerprint.py`) — si 07 está comprometido pero
su fingerprint no existe, falla explícitamente en vez de degradarse a
`None` en silencio.

## 20. Limitaciones actuales

- La ruta real de 07/08 (con LLM y Chroma reales) **no se ha ejecutado
  todavía** en este entorno de desarrollo — toda la migración se validó con
  dobles deterministas (LLM, retriever, embeddings, BERTScore). El smoke
  test real en Colab es el primer punto donde se ejercita la integración
  con red real.
- El gate determinista de precheck para claims **cuantitativos**
  (`numeric_risk`) está caracterizado (ver
  `tests/orchestration/test_verification_numeric_risk_characterization.py`)
  pero no se investigó su relación completa con el notebook 03B ni se
  intentó resolver — queda como tarea independiente.
- 07C permanece en el código por compatibilidad histórica pero no forma
  parte del registro de etapas activo; su eliminación física es una
  migración separada.

## 21. Estado de validación

- **468/468** escenarios en 37 suites de `tests/orchestration/`, ejecutados
  con `python3 archivo.py` (no `pytest`), verificados dos veces: en el
  entorno de desarrollo y descomprimiendo el ZIP de entrega en un
  directorio nuevo.
- El ciclo cualitativo completo `06 → 07 (RETURN) → 06 (REVISION) → 07
  (ADVANCE)` está probado de punta a punta con código productivo real
  (`VerificationAgent`, `propose_correction`,
  `build_provisional_verification_traceability_bundle`,
  `DraftWritingAgent` en modo `REVISION`), sustituyendo únicamente LLM,
  Chroma y retriever por dobles deterministas.
- La etapa 08 completa (métricas automáticas, factuales, LLM Judge,
  persistencia de los 15 outputs, fingerprints, contrato transaccional,
  `StageSpec`) está migrada y probada con las mismas sustituciones.

## 22. Advertencia: smoke test real con Chroma y OpenAI

**Nada de lo anterior sustituye una corrida real.** Todas las pruebas de
este repositorio usan dobles deterministas para LLM, Chroma y retriever —
ninguna ejercita la integración real de red, autenticación, límites de tasa
o comportamiento no determinista de un LLM real. El primer punto donde eso
se valida es el smoke test real descrito en `COLAB_SMOKE_TEST.md`, que debe
ejecutarse en un entorno con credenciales de OpenAI y una colección Chroma
real.

## 23. Reproducibilidad

- Todas las funciones puras de `src/tools/` no leen ni escriben archivos ni
  llaman a red — reciben sus datos de entrada y devuelven estructuras en
  memoria, lo que las hace deterministas y testeables sin dobles.
- `langdetect` fija `DetectorFactory.seed = 0` para detección de idioma
  determinista (ver `src/tools/evaluation/language_preprocessing.py`).
- Los fingerprints (sección 19) permiten verificar si una corrida
  reproduce exactamente las mismas entradas/configuración/dependencias que
  una corrida previa.
- El componente **no determinista real** es el propio LLM (OpenAI): el
  pipeline no fija `temperature=0` de forma global ni cachea respuestas de
  producción; la reproducibilidad exacta del texto generado no está
  garantizada entre corridas con LLM real.

## 24. Relación con la tesis

Este repositorio es el soporte de implementación de la tesis. El diseño
multiagente, el ciclo correctivo 06↔07, y las métricas de evaluación de la
etapa 08 corresponden directamente a los objetivos específicos y la
metodología descritos en el documento de tesis. Los notebooks operativos
originales (00-08) permanecen como fuente de referencia científica en
`notebooks_2/`; este repositorio contiene la migración a un sistema
transaccional productivo (`StateStore`, `StageSpec`, contratos
`AgentInput`/`AgentResult`) sobre esa misma base científica, sin alterar
las decisiones científicas de los notebooks originales salvo donde se
documenta explícitamente lo contrario (ver secciones 6 y 20).
