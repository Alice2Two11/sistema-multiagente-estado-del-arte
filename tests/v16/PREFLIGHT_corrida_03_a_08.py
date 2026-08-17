# =============================================================================
# PREFLIGHT fail-closed para Corrida_03_a_08 -- reemplaza las celdas 16 y 17
# del notebook actual.
#
# Las celdas 16-17 ACTUALES escriben active_experiment.json
# manualmente (activaban exclude_reviews=True y agregaban
# DOCUMENT_TYPE_UNKNOWN_AND_CARD_INVALID a allowed_reason_codes). Eso
# ya NO es necesario ni deseable: la política metodológica
# (exclude_reviews=True, corpus_eligibility_policy con
# min_include_corpus_size=1) ahora nace materializada por defecto en
# src/config/generation_policy_config.py -- la fuente canónica del
# sistema, no una celda de notebook.
#
# Este preflight NO escribe nada. Si algo no está en el estado
# esperado, DETIENE la corrida explícitamente (fail-closed) en vez de
# repararlo silenciosamente -- así cualquier desviación real (por
# ejemplo, alguien sobrescribiendo exclude_reviews=False a propósito
# en OTRO active_experiment.json, lo cual es válido y debe respetarse
# conscientemente, pero nunca para "Corrida_03_a_08" específicamente
# para experimentos de tesis) queda visible de inmediato, nunca oculta
# tras una reparación automática.
# =============================================================================

from pathlib import Path
import subprocess

PYTHON = Path("/content/venv_estado_arte/bin/python")
PROJECT = Path("/content/proyecto_estado_arte")

code = r'''
import json
import sys

import src
from src.config.generation_policy_config import get_extraction_policy

policy = get_extraction_policy()

print("=" * 100)
print("PREFLIGHT -- Política efectiva de Stage03 (solo lectura, nunca se escribe aquí)")
print("=" * 100)
print("src:", src.__file__)
print()
print("exclude_reviews:", policy.get("exclude_reviews"))
print("corpus_eligibility_policy:", policy.get("corpus_eligibility_policy"))

assert policy.get("exclude_reviews") is True, (
    "PREFLIGHT FALLIDO: extraction_policy.exclude_reviews no es True. "
    "Para experimentos de tesis, los reviews/surveys completos deben "
    "quedar excluidos del corpus de generación por metodología. Si "
    "esto es intencional (override consciente), detén esta corrida y "
    "no uses este preflight -- no lo repares aquí."
)

corpus_eligibility_policy = policy.get("corpus_eligibility_policy") or {}
min_include = corpus_eligibility_policy.get("min_include_corpus_size")
assert isinstance(min_include, int) and min_include >= 1, (
    "PREFLIGHT FALLIDO: corpus_eligibility_policy.min_include_corpus_size "
    f"no es un entero >= 1 (valor actual: {min_include!r})."
)

print()
print("PREFLIGHT OK -- exclude_reviews=True, corpus eligibility gate activo "
      f"(min_include_corpus_size={min_include}). Continuando sin modificar "
      "active_experiment.json.")
'''

result = subprocess.run(
    [str(PYTHON), "-c", code],
    cwd=str(PROJECT),
    text=True,
    capture_output=True,
)

print(result.stdout)
if result.stderr:
    print("\nSTDERR:")
    print(result.stderr)

assert result.returncode == 0, "PREFLIGHT FALLIDO -- ver salida arriba. La corrida se detiene."
print("\nRETURN CODE:", result.returncode)
