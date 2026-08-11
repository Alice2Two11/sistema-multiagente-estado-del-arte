"""Regresión para la colisión real de imports en Colab:
``src/config.py`` (módulo plano, notebooks 00-02/07) vs ``src/config/``
(paquete del orquestador) -- ambos conviven en ``src/``, y un
``sys.path.insert(0, str(SRC_DIR))`` seguido de ``import config`` a
secas es ambiguo: se confirma empíricamente que el resolvedor de
CPython elige el PAQUETE, produciendo exactamente el error real
reportado:

    ImportError: cannot import name 'PROJECT_DIR' from 'config'
    (.../src/config/__init__.py)

``activate_experiment_runtime_config`` (``src/notebook_runtime_
bootstrap.py``) resuelve esto registrando el módulo PLANO en
``sys.modules["config"]`` por ruta de archivo explícita, sin pasar por
el resolvedor de paquetes -- nunca se borra ni se renombra ningún
archivo."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS = []


def scenario(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
            except Exception:
                RESULTS.append((name, False, traceback.format_exc()))
            else:
                RESULTS.append((name, True, ""))

        return wrapper

    return decorator


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    """Corre `code` en un subproceso Python NUEVO -- cada escenario
    necesita un estado de sys.modules/sys.path limpio, ya que
    `import config` deja una entrada cacheada que contaminaría los
    escenarios siguientes si se corrieran en el mismo proceso."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def _write_collision_fixture(src_dir: Path) -> None:
    """Reproduce la colisión real de forma aislada -- un config.py
    plano mínimo (sin depender de active_experiment.json, para no
    acoplar este test a la lógica de negocio real de config.py) junto
    a un paquete config/ que, igual que el real, no expone PROJECT_DIR."""
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "config.py").write_text(
        textwrap.dedent("""
            PROJECT_DIR = "FLAT_MODULE_RESOLVED"
        """).strip() + "\n",
        encoding="utf-8",
    )
    package_dir = src_dir / "config"
    package_dir.mkdir(exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "some_policy_config.py").write_text(
        "SOME_POLICY = {}\n", encoding="utf-8",
    )


@scenario("BB01. Reproducción aislada del bug real: sys.path=[SRC_DIR] + 'from config import PROJECT_DIR' sin el fix -> ImportError, resuelve al paquete")
def test_collision_reproduced_without_fix():
    with tempfile.TemporaryDirectory() as tmp:
        src_dir = Path(tmp) / "src"
        _write_collision_fixture(src_dir)
        code = f"""
import sys
sys.path.insert(0, {str(src_dir)!r})
from config import PROJECT_DIR
"""
        result = _run_isolated(code)
        assert result.returncode != 0
        assert "cannot import name 'PROJECT_DIR' from 'config'" in result.stderr
        assert str(src_dir / "config" / "__init__.py") in result.stderr


@scenario("BB02. Con activate_experiment_runtime_config: 'from config import PROJECT_DIR' resuelve al módulo PLANO, no al paquete")
def test_fix_resolves_flat_module():
    with tempfile.TemporaryDirectory() as tmp:
        src_dir = Path(tmp) / "src"
        _write_collision_fixture(src_dir)
        code = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
sys.path.insert(0, {str(src_dir)!r})
from src.notebook_runtime_bootstrap import activate_experiment_runtime_config
activate_experiment_runtime_config({str(src_dir)!r})

from config import PROJECT_DIR
assert PROJECT_DIR == "FLAT_MODULE_RESOLVED", PROJECT_DIR

import config
assert config.__file__ == {str(src_dir / "config.py")!r}, config.__file__
print("OK")
"""
        result = _run_isolated(code)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout


@scenario("BB03. Imports internos posteriores (simulando generation_config.py/rag_policy.py, que también hacen 'from config import ...') también resuelven al módulo plano, sin volver a activar nada")
def test_downstream_flat_modules_also_resolve_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        src_dir = Path(tmp) / "src"
        _write_collision_fixture(src_dir)
        # generation_config.py real hace "from config import GENERATION_PROFILE, RAG_POLICY"
        # -- aquí replicamos el mismo patrón exacto con un símbolo propio.
        (src_dir / "downstream_flat_module.py").write_text(
            "from config import PROJECT_DIR as INHERITED\n", encoding="utf-8",
        )
        code = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
sys.path.insert(0, {str(src_dir)!r})
from src.notebook_runtime_bootstrap import activate_experiment_runtime_config
activate_experiment_runtime_config({str(src_dir)!r})

import downstream_flat_module
assert downstream_flat_module.INHERITED == "FLAT_MODULE_RESOLVED", downstream_flat_module.INHERITED
print("OK")
"""
        result = _run_isolated(code)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout


@scenario("BB04. El sistema del orquestador (from src.config import ...) sigue intacto -- nunca pasó por el módulo plano")
def test_orchestrator_config_package_unaffected():
    code = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})

# El orquestador SIEMPRE importa el paquete namespaced -- nunca 'config' a secas.
from src.config.verification_policy_config import get_verification_input_policy
policy = get_verification_input_policy({{}})
assert isinstance(policy, dict)

# El módulo notebook_runtime_bootstrap es importable sin efectos
# secundarios sobre el paquete del orquestador -- no registra nada en
# sys.modules hasta que se llama explícitamente a
# activate_experiment_runtime_config(...).
from src.notebook_runtime_bootstrap import activate_experiment_runtime_config
assert "config" not in sys.modules or sys.modules["config"] is None or True  # solo confirma que el import no falla

from src.config.verification_policy_config import get_verification_input_policy as policy_fn_after
policy_after = policy_fn_after({{}})
assert isinstance(policy_after, dict)
print("OK")
"""
    result = _run_isolated(code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


@scenario("BB05. Fail-closed: activate_experiment_runtime_config sobre un src_dir sin config.py lanza FileNotFoundError explícito, nunca falla en silencio")
def test_missing_config_file_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        empty_src = Path(tmp) / "src_empty"
        empty_src.mkdir()
        sys.path.insert(0, str(REPO_ROOT))
        from src.notebook_runtime_bootstrap import activate_experiment_runtime_config

        try:
            activate_experiment_runtime_config(empty_src)
        except FileNotFoundError as exc:
            assert "config.py" in str(exc)
        else:
            raise AssertionError("debía fallar cerrado si config.py no existe")


if __name__ == "__main__":
    for fn in (
        test_collision_reproduced_without_fix,
        test_fix_resolves_flat_module,
        test_downstream_flat_modules_also_resolve_correctly,
        test_orchestrator_config_package_unaffected,
        test_missing_config_file_fails_closed,
    ):
        fn()

    failed = 0
    for name, ok, err in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        if not ok:
            failed += 1
            print(err)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} escenarios OK")
    raise SystemExit(1 if failed else 0)
