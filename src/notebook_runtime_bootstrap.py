"""Activación formal del runtime de configuración plana para notebooks.

COLISIÓN REAL: ``src/config.py`` (módulo plano, escrito y leído por los
notebooks operativos 00/01/02/07 vía ``from config import ...``) y
``src/config/`` (paquete del sistema del orquestador, con las políticas
por etapa que consume ``src/adapters``/``src/agents``/``src/tools`` vía
``from src.config import ...``) conviven, a propósito, en el mismo
directorio -- ninguno de los dos se elimina ni se renombra (ver README,
sección 26).

Cuando un notebook hace ``sys.path.insert(0, str(SRC_DIR))`` (poniendo
``.../src`` directamente en el path, el patrón real de los notebooks
00-02/07) y luego ``import config`` / ``from config import PROJECT_DIR``
a secas, la búsqueda es AMBIGUA: ``src/`` contiene tanto ``config.py``
como ``config/``. El resolvedor de import de CPython, al encontrar ambos
en el mismo directorio, resuelve al PAQUETE (``config/__init__.py``) --
confirmado empíricamente, no es una suposición -- y por eso el notebook
falla con::

    ImportError: cannot import name 'PROJECT_DIR' from 'config'
    (.../src/config/__init__.py)

Esto nunca afecta al sistema del orquestador: ese código SIEMPRE importa
el paquete como ``src.config`` (namespaced bajo ``src.``), nunca como
``config`` a secas -- por eso los 574 tests de ``tests/orchestration/``
nunca lo manifestaron.

SOLUCIÓN (permanente, sin borrar ni renombrar ningún archivo): registrar
explícitamente el módulo PLANO en ``sys.modules["config"]`` ANTES de que
cualquier ``import config``/``from config import ...`` ambiguo se
ejecute -- vía ``importlib.util.spec_from_file_location``, apuntando
directamente al archivo ``config.py``, sin pasar por el resolvedor de
paquetes en absoluto. Una vez registrado, TODO ``import config``
posterior -- incluido el que hacen internamente ``generation_config.py``,
``rag_policy.py``, ``experiment_config.py`` y ``rag_utils.py`` al
ejecutarse -- encuentra la entrada ya cacheada en ``sys.modules`` y usa
el módulo plano, sin volver a consultar el resolvedor.

Uso en un notebook (00, 01, 02 o 07), inmediatamente después de insertar
``SRC_DIR`` en ``sys.path`` y ANTES de cualquier ``from config import
...``::

    sys.path.insert(0, str(SRC_DIR))

    sys.path.insert(0, str(SRC_DIR))
    from notebook_runtime_bootstrap import activate_experiment_runtime_config
    activate_experiment_runtime_config(SRC_DIR)

    from config import PROJECT_DIR  # ahora resuelve al módulo plano, siempre
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def activate_experiment_runtime_config(src_dir: str | Path, *, module_name: str = "config") -> ModuleType:
    """Carga ``<src_dir>/config.py`` (el módulo PLANO, nunca el paquete
    ``config/``) por ruta de archivo explícita -- sin pasar por el
    resolvedor de paquetes de Python, así que la ambigüedad con
    ``src/config/`` nunca se produce -- y lo registra en
    ``sys.modules[module_name]`` (``"config"`` por defecto).

    A partir de esta llamada, cualquier ``import config`` / ``from
    config import X`` en el mismo proceso (incluido el que ejecutan
    internamente otros módulos planos al importarse, como ``rag_policy.py``
    o ``experiment_config.py``) encuentra la entrada ya cacheada y usa
    este módulo -- nunca vuelve a resolver el nombre ``config`` de forma
    ambigua.

    Idempotente: si se llama más de una vez, simplemente reejecuta
    ``config.py`` de nuevo (útil tras editar el archivo) y actualiza la
    misma entrada de ``sys.modules`` -- no acumula módulos duplicados.

    Falla cerrado (``FileNotFoundError``/``ImportError`` reales, nunca en
    silencio) si ``<src_dir>/config.py`` no existe o si su ejecución
    lanza una excepción -- nunca se enmascara un error real de
    configuración."""

    src_dir = Path(src_dir)
    config_path = src_dir / "config.py"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No existe {config_path} -- activate_experiment_runtime_config "
            "requiere el módulo plano de configuración ya escrito (ver "
            "00_setup_config.ipynb, celda que hace %%writefile .../src/config.py)."
        )

    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo construir un spec de import para {config_path}.")

    module = importlib.util.module_from_spec(spec)
    # Se registra ANTES de ejecutar el módulo (mismo orden que usa el
    # import real de Python) para que imports circulares/recursivos
    # dentro de config.py, si los hubiera, también encuentren la entrada.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
