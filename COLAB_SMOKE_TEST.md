# Smoke test real en Colab — 06 → 07 ↔ 06 → 07 → 08

Este documento describe el procedimiento para validar, con OpenAI y Chroma
**reales** (no dobles deterministas), el ciclo completo del pipeline. Todo
lo que se probó en el entorno de desarrollo de esta entrega usó dobles
deterministas — este es el primer punto donde se ejercita la integración
real de red, autenticación y comportamiento no determinista del LLM.

## 1. Descomprimir el proyecto

```bash
!unzip -q tesis-sistema-multiagente-ACUMULADO-completo.zip -d /content
%cd /content/tesis-sistema-multiagente-main
```

Verificar que la extracción trajo la estructura esperada:

```bash
!ls src/tools/evaluation/ src/orchestration/
```

Debe listar, entre otros, `evaluation_pipeline.py` (en
`src/tools/evaluation/`) y `pipeline_orchestrator.py` (en
`src/orchestration/`). Si alguno falta, el ZIP no es el acumulado completo
— no continuar.

## 2. Instalar dependencias

```bash
!pip install -q -r requirements.txt
```

`requirements.txt` distingue qué versiones se verificaron localmente
(pandas, numpy, PyMuPDF, langdetect, rouge-score, sentence-transformers,
scikit-learn, tabulate, cryptography) de las que **no** se probaron en el
entorno de desarrollo por requerir red real (`langchain-openai`, `openai`,
`chromadb`, `bert-score`) — presta atención a errores de instalación en
estas últimas particularmente.

## 3. Configurar `OPENAI_API_KEY`

En Colab, la forma recomendada es usar Secrets del notebook (icono de
llave en la barra lateral) con el nombre `OPENAI_API_KEY`, y luego:

```python
from google.colab import userdata
import os
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

`src/io/credentials.py` también acepta la variable de entorno directamente
si se prefiere:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

Verificar que quedó disponible:

```python
import os
assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY no está configurada"
```

## 4. Definir `PROJECT_DIR`

`PROJECT_DIR` es la carpeta que contiene `active_experiment.json` y, dentro
de ella, la carpeta del experimento activo (`PROJECT_DIR/{experiment_id}/`)
con los artefactos ya producidos por las etapas 00-05 (y, si se quiere
retomar desde 06 o 07, también los de esas etapas).

```python
PROJECT_DIR = "/content/drive/MyDrive/tesis/experimentos/exp_paper_02"
```

Debe apuntar a Google Drive montado (o a una ruta persistente equivalente)
— **no** dejarlo dentro de `/content` sin respaldo, porque `/content` es
efímero y se pierde al cerrar la sesión de Colab.

## 5. Validar `active_experiment.json`

```python
import json
from pathlib import Path

active_path = Path(PROJECT_DIR) / "active_experiment.json"
assert active_path.is_file(), f"No existe {active_path} -- corre primero la etapa 00"

active = json.loads(active_path.read_text(encoding="utf-8"))
assert "active_experiment_id" in active, "active_experiment.json debe tener 'active_experiment_id'"
assert isinstance(active.get("evaluation_policy"), dict) and active["evaluation_policy"], (
    "active_experiment.json debe tener 'evaluation_policy' como diccionario no vacío "
    "-- lo exige build_execution_for_stagespec() para la etapa 08."
)
print(active)
```

Formato mínimo esperado (ver `README.md` sección 12 para el detalle
completo):

```json
{
  "active_experiment_id": "exp_paper_02",
  "run_id": "run_2026_08_07"
}
```

## 6. Ejecutar primero hasta 07 (aísla el ciclo 06↔07 de la etapa 08)

```bash
!python3 -m src.orchestration.pipeline_orchestrator \
  --project-dir "$PROJECT_DIR" \
  --until 07_agente_verificador \
  --force-rerun
```

`--force-rerun` fuerza a reejecutar la etapa inicial aunque ya tenga un
resultado `COMPLETED` vigente en `pipeline_state.json` — útil para repetir
el smoke test sin borrar el estado manualmente.

**Qué revisar si esto falla**: el traceback completo, y si el proceso
avanzó al menos hasta invocar el LLM real (para distinguir un error de
configuración/credenciales de un error de integración con Chroma o de
lógica). Ver sección 9 sobre qué conservar.

**Si 07 emite `RETURN`**: el mismo comando, ejecutado de nuevo sin
`--force-rerun`, retoma el ciclo — el orquestador interpreta la transición
real y vuelve a 06 automáticamente. No hace falta invocarlo etapa por
etapa.

## 7. Ejecutar después hasta el pipeline completo (incluye 08)

Una vez que el paso anterior deja el ciclo 06↔07 resuelto (07 en
`ADVANCE`), correr sin `--until` para completar hasta 08 y
`STOP_PIPELINE`:

```bash
!python3 -m src.orchestration.pipeline_orchestrator \
  --project-dir "$PROJECT_DIR" \
  --force-rerun
```

Si se prefiere forzar todo desde cero en un solo comando (sin el paso 6
por separado), este mismo comando ya cubre `06 → 07 ↔ 06 → 07 → 08 → fin`
— el paso 6 existe para poder diagnosticar el ciclo 06↔07 de forma aislada
antes de involucrar la etapa 08 (más lenta, por las descargas de modelos
de `sentence-transformers`/`bert-score`).

## 8. Localizar los artefactos generados

### `pipeline_state.json`

```python
state_path = Path(PROJECT_DIR) / active["active_experiment_id"] / "05_outputs" / "00_orchestrator_planner" / "pipeline_state.json"
print(state_path, state_path.is_file())
```

Revisar en él: qué etapas quedaron `COMPLETED`/`FAILED`, sus fingerprints,
y el `AgentResult` persistido de cada una.

### `writer_verifier_cycle/round_01` (y rondas siguientes, si hubo más de una)

```python
cycle_dir = Path(PROJECT_DIR) / active["active_experiment_id"] / "05_outputs" / "writer_verifier_cycle"
for round_dir in sorted(cycle_dir.glob("round_*")):
    print(round_dir, [p.name for p in round_dir.iterdir()])
```

Cada ronda debe tener, como mínimo: `writer_revision_request.json`,
`_round_status.json` (revisar su campo de estado:
`AWAITING_REVISION` → `REVISION_COMPLETED`), y — una vez que 06 la
completa — `revised_draft.json`, `revision_changelog.json`,
`revision_resolution_matrix.json`.

### Outputs de 07

```python
verify_dir = Path(PROJECT_DIR) / active["active_experiment_id"] / "05_outputs" / "06_verification_traceability"
print(list(verify_dir.iterdir()))
```

**Nota sobre el nombre de la carpeta**: es `06_verification_traceability`,
no `07_verification` — el nombre real de carpeta hereda la numeración del
notebook original, no la del `StageSpec` (que usa la clave
`07_agente_verificador`). Confirmado leyendo
`src/adapters/verification_orchestrator_runtime.py`.

Deben existir los 4 artefactos científicos incondicionales
(`provisional_verification_traceability_bundle.json`,
`multi_proposal_resolution_result.json`, `agent07_runtime_report.json`,
`agent07_artifact_manifest.json`), más `writer_revision_request.json`
**solo** si la última transición comprometida fue `RETURN`.

## 9. Comprobar los 15 outputs de 08

```python
eval_dir = Path(PROJECT_DIR) / active["active_experiment_id"] / "05_outputs" / "07_evaluation"

required = [
    "automatic_metrics.csv", "semantic_chunk_alignment.csv",
    "bertscore_chunk_alignment.csv", "factual_metrics.csv",
    "final_citation_check.csv", "final_claim_audit.csv",
    "llm_judge_evaluation.json", "llm_judge_scores.csv",
    "corpus_gap_suggestions.csv", "corpus_gap_suggestions.md",
    "final_selected_metrics.csv", "evaluation_summary.json",
    "final_evaluation_report.md", "evaluation_validation_report.json",
    "evaluation_manifest.json",
]
missing = [name for name in required if not (eval_dir / name).is_file()]
print("faltantes:", missing or "ninguno -- los 15 outputs están completos")
```

**Nota sobre el nombre de la carpeta**: es `07_evaluation`, no
`08_evaluacion_experimental` — mismo motivo que en 07: nombre heredado del
notebook, confirmado leyendo
`src/adapters/evaluation_stagespec_wiring.py`
(`dir_evaluation = outputs / "07_evaluation"`).

`agent08_upstream_numeric_check.csv`, si aparece en esa carpeta, es un
artefacto **intermedio** (no cuenta como uno de los 15) — ver `README.md`
sección 15.

## 10. Qué archivos conservar si ocurre un error

Antes de reintentar o modificar nada, copiar (por ejemplo a otra carpeta de
Drive) para poder diagnosticar sin perder evidencia:

```text
1. El traceback completo de la consola (copiar el texto, no solo el resumen).
2. pipeline_state.json completo.
3. Toda la carpeta writer_verifier_cycle/ (todas las rondas, no solo la última).
4. La carpeta 07_verification/ completa (incluye el manifest con hashes).
5. Lo que exista de 08_evaluacion_experimental/ aunque esté incompleto.
6. active_experiment.json (para confirmar qué experimento/run se corrió).
```

Con eso, la siguiente ronda de revisión puede distinguir entre un error de
configuración/credenciales, un problema de integración con Chroma/OpenAI, y
un defecto real de código — sin necesidad de reproducir el fallo de nuevo
en Colab primero.
