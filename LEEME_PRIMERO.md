# LÉEME PRIMERO — guía corta y operativa

Para la documentación técnica completa (arquitectura, flujo detallado,
formatos, outputs, reproducibilidad, pruebas) ver **`README.md`**.

## ¿Qué hace el sistema?

Genera, verifica y evalúa automáticamente **estados del arte** (revisiones
de literatura científica) a partir de un corpus de artículos, usando un
pipeline multiagente basado en LLM.

El sistema tiene **dos capas** que cooperan, no una cadena única de
etapas idénticas:

- **Capa A — preparación del experimento**: notebooks `00_setup_config`,
  `01_ingesta_memoria_documental`, `02_rag_chroma_retriever`. Se ejecutan
  manualmente, en orden, sin estado transaccional.
- **Capa B — pipeline científico**: `Corrida_03_a_08.ipynb`, que invoca a
  `src.orchestration.pipeline_orchestrator` para ejecutar
  `03 → 03B → 04 → 05 → 06 ↔ 07 → 08` con estado persistente y
  reintentos reales.

## ¿Qué ejecuto primero?

```text
1. 00_setup_config.ipynb
2. 01_ingesta_memoria_documental.ipynb
3. 02_rag_chroma_retriever.ipynb
4. Corrida_03_a_08.ipynb
```

## ¿Qué hacen 00, 01 y 02?

- **00**: crea o retoma el experimento, fija tema/alcance/idioma/estilo,
  y deja escrito `active_experiment.json` (el selector del experimento
  activo que todo lo demás lee).
- **01**: valida los PDFs, mantiene el Ground Truth en un directorio
  separado desde el inicio (nunca se mezcla con el corpus de referencia),
  extrae texto, genera chunks, y produce `chunks_clean_for_rag` — solo a
  partir de los papers de referencia.
- **02**: carga `chunks_clean_for_rag`, valida que no contenga Ground
  Truth, y construye la colección Chroma que la Capa B usará para
  recuperar evidencia.

## ¿Qué hace `Corrida_03_a_08.ipynb`?

Sincroniza el código desde GitHub (preservando experimentos y el
experimento activo), prepara el entorno, identifica el experimento
activo, y llama al orquestador real para ejecutar 03→08. Al final,
resume el resultado y **consolida** (sin volver a generar ciencia ni
modificar resultados) los artefactos ya producidos en archivos listos
para la tesis.

Detalle completo, incluida la distinción entre `approved_for_
publication` y `usable_for_evaluation` tras un `PARTIAL_HALT` de
verificación, en `README.md`.

## ¿Dónde quedan los resultados?

```text
{experimento}/05_outputs/06_verification_traceability/   -> artefactos de verificación
{experimento}/05_outputs/07_evaluation/                   -> los 15 outputs de evaluación
{experimento}/05_outputs/08_final_results/                -> consolidación final para tesis
```

## ¿Dónde está la documentación completa?

En **`README.md`**: arquitectura completa, Corpus Eligibility Gate,
separación de Ground Truth, RAG, ciclo correctivo 06↔07, evaluación 08,
formato de `active_experiment.json`, instalación, pruebas y
reproducibilidad.
