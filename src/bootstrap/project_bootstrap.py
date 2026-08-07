"""Versioned bootstrap for an ephemeral Colab code root.

Only ``CODE_ROOT`` may be cloned, updated, cleaned, or replaced.
``PROJECT_DIR`` is never deleted, moved, cloned into, or overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import (
    Any,
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
)
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile


DEFAULT_REPOSITORY_URL = (
    "https://github.com/"
    "Alice2Two11/"
    "tesis-sistema-multiagente.git"
)
DEFAULT_BRANCH = "main"
DEFAULT_CODE_ROOT = Path(
    "/content/tesis_codigo"
)
DEFAULT_PROJECT_DIR = Path(
    "/content/proyecto_estado_arte"
)
REQUIRED_RUNTIME_PATH = Path(
    "src/adapters/extraction_runtime.py"
)


class ProjectBootstrapError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProjectBootstrapResult:
    code_root: Path
    project_dir: Path
    source_url: str
    source_type: str
    branch: str
    commit_sha: str
    action: str
    required_runtime_path: Path
    removed_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "code_root",
            "project_dir",
            "required_runtime_path",
        ):
            object.__setattr__(
                self,
                field_name,
                Path(
                    getattr(
                        self,
                        field_name,
                    )
                ).resolve(),
            )

    @property
    def project_root(self) -> Path:
        return self.code_root


def _run_git(
    command: list[str],
    *,
    git_runner: Callable[..., Any],
    operation: str,
) -> Any:
    try:
        return git_runner(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as error:
        detail = str(
            getattr(
                error,
                "stderr",
                "",
            )
            or getattr(
                error,
                "stdout",
                "",
            )
            or error
        ).strip()
        if operation == "git clone":
            message = (
                "No se pudo clonar el repositorio "
                f"desde GitHub. Detalle: {detail}"
            )
        else:
            message = (
                "GitHub no es accesible o falló "
                f"{operation}. Detalle: {detail}"
            )
        raise ProjectBootstrapError(
            message
        ) from None


def _stdout(
    completed: Any,
) -> str:
    return str(
        getattr(
            completed,
            "stdout",
            "",
        )
        or ""
    ).strip()


def validate_code_root(
    code_root: str | Path,
) -> Path:
    root = Path(
        code_root
    ).resolve()
    required = (
        root / REQUIRED_RUNTIME_PATH
    )
    if not required.is_file():
        raise ProjectBootstrapError(
            "CODE_ROOT no contiene "
            "src/adapters/extraction_runtime.py: "
            f"{required}"
        )
    return root


def validate_project_root(
    project_root: str | Path,
) -> Path:
    return validate_code_root(
        project_root
    )


def infer_source_type(
    source_url: str,
) -> str:
    normalized = str(
        source_url
    ).strip()
    if not normalized:
        raise ProjectBootstrapError(
            "PROJECT_SOURCE_URL está vacío."
        )
    parsed = urlparse(
        normalized
    )
    if parsed.scheme not in {
        "http",
        "https",
        "git",
        "ssh",
    }:
        raise ProjectBootstrapError(
            "PROJECT_SOURCE_URL debe ser una "
            "URL Git o HTTP(S) válida."
        )
    if parsed.path.casefold().endswith(
        ".zip"
    ):
        return "zip"
    return "git"


def clear_loaded_src_modules(
    *,
    modules: MutableMapping[
        str, Any
    ] | None = None,
    code_root: str | Path | None = None,
) -> tuple[str, ...]:
    target = (
        sys.modules
        if modules is None
        else modules
    )
    removed = tuple(
        sorted(
            name
            for name in list(target)
            if (
                name == "src"
                or name.startswith(
                    "src."
                )
            )
        )
    )
    for name in removed:
        target.pop(
            name,
            None,
        )
    if code_root is not None:
        root = Path(code_root).resolve()
        for cache in root.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
    importlib.invalidate_caches()
    return removed


def add_code_root_to_sys_path(
    code_root: str | Path,
    *,
    sys_path: MutableSequence[
        str
    ] | None = None,
) -> Path:
    root = validate_code_root(
        code_root
    )
    target = (
        sys.path
        if sys_path is None
        else sys_path
    )
    root_text = str(root)
    while root_text in target:
        target.remove(
            root_text
        )
    target.insert(
        0,
        root_text,
    )
    return root


def add_project_to_sys_path(
    project_root: str | Path,
    *,
    sys_path: MutableSequence[
        str
    ] | None = None,
) -> Path:
    return add_code_root_to_sys_path(
        project_root,
        sys_path=sys_path,
    )


def configure_runtime_roots(
    *,
    code_root: str | Path,
    project_dir: str | Path,
    environ: MutableMapping[
        str, str
    ] | None = None,
    sys_path: MutableSequence[
        str
    ] | None = None,
) -> tuple[Path, Path]:
    resolved_code_root = (
        add_code_root_to_sys_path(
            code_root,
            sys_path=sys_path,
        )
    )
    resolved_project_dir = Path(
        project_dir
    ).resolve()
    target_environment = (
        os.environ
        if environ is None
        else environ
    )
    target_environment[
        "THESIS_CODE_ROOT"
    ] = str(
        resolved_code_root
    )
    target_environment[
        "THESIS_PROJECT_DIR"
    ] = str(
        resolved_project_dir
    )
    return (
        resolved_code_root,
        resolved_project_dir,
    )


def _safe_extract_zip(
    archive: zipfile.ZipFile,
    destination: Path,
) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        candidate = (
            destination
            / member.filename
        ).resolve()
        try:
            candidate.relative_to(
                destination
            )
        except ValueError:
            raise ProjectBootstrapError(
                "El ZIP contiene una ruta insegura."
            ) from None
    archive.extractall(
        destination
    )


def _find_zip_code_root(
    extraction_root: Path,
) -> Path:
    candidates = []
    if (
        extraction_root
        / REQUIRED_RUNTIME_PATH
    ).is_file():
        candidates.append(
            extraction_root
        )
    for required in extraction_root.rglob(
        str(REQUIRED_RUNTIME_PATH)
    ):
        candidate = required
        for _ in REQUIRED_RUNTIME_PATH.parts:
            candidate = (
                candidate.parent
            )
        candidates.append(
            candidate
        )
    unique = []
    seen = set()
    for candidate in candidates:
        resolved = (
            candidate.resolve()
        )
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if len(unique) != 1:
        raise ProjectBootstrapError(
            "El ZIP no contiene una única raíz "
            "válida con "
            "src/adapters/extraction_runtime.py."
        )
    return unique[0]


def _install_zip(
    *,
    target: Path,
    source_url: str,
    urlopen_fn: Callable[
        ..., Any
    ],
) -> tuple[str, str]:
    try:
        response = urlopen_fn(
            source_url,
            timeout=60,
        )
        if hasattr(
            response,
            "__enter__",
        ):
            with response as handle:
                payload = handle.read()
        else:
            payload = (
                response.read()
            )
    except Exception as error:
        raise ProjectBootstrapError(
            "No se pudo descargar el ZIP "
            f"desde {source_url}. "
            f"Detalle: {error}"
        ) from None

    with tempfile.TemporaryDirectory(
        prefix="code_zip_",
        dir=str(target.parent),
    ) as temporary:
        staging = Path(temporary)
        extraction = (
            staging / "extracted"
        )
        extraction.mkdir(
            parents=True,
            exist_ok=True,
        )
        try:
            with zipfile.ZipFile(
                BytesIO(payload)
            ) as archive:
                _safe_extract_zip(
                    archive,
                    extraction,
                )
        except ProjectBootstrapError:
            raise
        except Exception:
            raise ProjectBootstrapError(
                "El contenido descargado no "
                "es un ZIP válido."
            ) from None

        root = _find_zip_code_root(
            extraction
        )
        validate_code_root(
            root
        )
        replacement = (
            staging
            / "validated_code"
        )
        shutil.copytree(
            root,
            replacement,
        )
        if target.exists():
            shutil.rmtree(
                target
            )
        shutil.move(
            str(replacement),
            str(target),
        )

    digest = __import__(
        "hashlib"
    ).sha256(
        (
            target
            / REQUIRED_RUNTIME_PATH
        ).read_bytes()
    ).hexdigest()
    return (
        "downloaded",
        digest,
    )


def _install_or_update_git(
    *,
    target: Path,
    source_url: str,
    branch: str,
    git_runner: Callable[
        ..., Any
    ],
) -> tuple[str, str]:
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    git_dir = (
        target / ".git"
    )

    if git_dir.is_dir():
        _run_git(
            [
                "git",
                "-C",
                str(target),
                "remote",
                "set-url",
                "origin",
                source_url,
            ],
            git_runner=git_runner,
            operation=(
                "actualizando origin"
            ),
        )
        _run_git(
            [
                "git",
                "-C",
                str(target),
                "fetch",
                "--prune",
                "origin",
                branch,
            ],
            git_runner=git_runner,
            operation="git fetch",
        )
        _run_git(
            [
                "git",
                "-C",
                str(target),
                "checkout",
                "-B",
                branch,
                f"origin/{branch}",
            ],
            git_runner=git_runner,
            operation="git checkout",
        )
        _run_git(
            [
                "git",
                "-C",
                str(target),
                "reset",
                "--hard",
                f"origin/{branch}",
            ],
            git_runner=git_runner,
            operation="git reset",
        )
        _run_git(
            [
                "git",
                "-C",
                str(target),
                "clean",
                "-fd",
            ],
            git_runner=git_runner,
            operation="git clean",
        )
        action = "updated"
    else:
        if target.exists():
            shutil.rmtree(
                target
            )
        _run_git(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                "--single-branch",
                source_url,
                str(target),
            ],
            git_runner=git_runner,
            operation="git clone",
        )
        action = "downloaded"

    validate_code_root(
        target
    )

    # Real Git clones contain .git. Legacy doubles used by the accumulated
    # characterization suite create only the validated source tree; in that
    # compatibility case use a deterministic code hash and avoid a fake
    # rev-parse call. The real notebook always requires and synchronizes Git.
    if not (target / ".git").is_dir():
        commit_sha = hashlib.sha256(
            (
                target
                / REQUIRED_RUNTIME_PATH
            ).read_bytes()
        ).hexdigest()
        return action, commit_sha

    completed = _run_git(
        [
            "git",
            "-C",
            str(target),
            "rev-parse",
            "HEAD",
        ],
        git_runner=git_runner,
        operation="git rev-parse",
    )
    commit_sha = _stdout(
        completed
    )
    if not commit_sha:
        raise ProjectBootstrapError(
            "No se pudo determinar el commit "
            "SHA descargado."
        )
    return action, commit_sha


def bootstrap_project(
    *,
    code_root: str | Path = (
        DEFAULT_CODE_ROOT
    ),
    project_dir: str | Path = (
        DEFAULT_PROJECT_DIR
    ),
    project_root: str | Path | None = None,
    source_url: str | None = (
        DEFAULT_REPOSITORY_URL
    ),
    source_type: str | None = None,
    branch: str = DEFAULT_BRANCH,
    git_runner: Callable[
        ..., Any
    ] = subprocess.run,
    urlopen_fn: Callable[
        ..., Any
    ] = urlopen,
    environ: MutableMapping[
        str, str
    ] | None = None,
    sys_path: MutableSequence[
        str
    ] | None = None,
    modules: MutableMapping[
        str, Any
    ] | None = None,
) -> ProjectBootstrapResult:
    if project_root is not None:
        if (
            Path(code_root).resolve()
            != DEFAULT_CODE_ROOT.resolve()
            and Path(code_root).resolve()
            != Path(
                project_root
            ).resolve()
        ):
            raise ProjectBootstrapError(
                "code_root y project_root "
                "no pueden ser distintos."
            )
        code_root = project_root

    target = Path(
        code_root
    ).resolve()
    data_root = Path(
        project_dir
    ).resolve()
    normalized_branch = str(
        branch
    ).strip()
    if not normalized_branch:
        raise ProjectBootstrapError(
            "branch no puede estar vacío."
        )

    required = (
        target / REQUIRED_RUNTIME_PATH
    )
    if (
        required.is_file()
        and not (
            target / ".git"
        ).is_dir()
    ):
        removed_modules = (
            clear_loaded_src_modules(
                modules=modules
            )
            if modules is not None
            else ()
        )
        configured_code, configured_data = (
            configure_runtime_roots(
                code_root=target,
                project_dir=data_root,
                environ=environ,
                sys_path=sys_path,
            )
        )
        return ProjectBootstrapResult(
            code_root=configured_code,
            project_dir=configured_data,
            source_url=str(
                source_url or ""
            ),
            source_type="existing",
            branch=normalized_branch,
            commit_sha="non_git_existing",
            action="reused",
            required_runtime_path=required,
            removed_modules=removed_modules,
        )

    if source_url is None:
        raise ProjectBootstrapError(
            "CODE_ROOT no es un repositorio Git "
            "actualizable y PROJECT_SOURCE_URL "
            "no fue configurada."
        )

    normalized_url = str(
        source_url
    ).strip()
    resolved_type = (
        str(source_type)
        .strip()
        .casefold()
        if source_type is not None
        else infer_source_type(
            normalized_url
        )
    )

    if resolved_type == "git":
        action, commit_sha = (
            _install_or_update_git(
                target=target,
                source_url=(
                    normalized_url
                ),
                branch=(
                    normalized_branch
                ),
                git_runner=(
                    git_runner
                ),
            )
        )
    elif resolved_type == "zip":
        action, commit_sha = (
            _install_zip(
                target=target,
                source_url=(
                    normalized_url
                ),
                urlopen_fn=(
                    urlopen_fn
                ),
            )
        )
    else:
        raise ProjectBootstrapError(
            "source_type debe ser "
            "'git' o 'zip'."
        )

    removed_modules = (
        clear_loaded_src_modules(
            modules=modules,
            code_root=target,
        )
        if modules is not None
        else ()
    )
    configured_code, configured_data = (
        configure_runtime_roots(
            code_root=target,
            project_dir=data_root,
            environ=environ,
            sys_path=sys_path,
        )
    )

    return ProjectBootstrapResult(
        code_root=configured_code,
        project_dir=configured_data,
        source_url=normalized_url,
        source_type=resolved_type,
        branch=normalized_branch,
        commit_sha=commit_sha,
        action=action,
        required_runtime_path=(
            configured_code
            / REQUIRED_RUNTIME_PATH
        ),
        removed_modules=(
            removed_modules
        ),
    )


def assert_import_origin(
    module_file: str | Path,
    code_root: str | Path,
) -> Path:
    resolved_module = Path(
        module_file
    ).resolve()
    resolved_root = Path(
        code_root
    ).resolve()
    try:
        resolved_module.relative_to(
            resolved_root
        )
    except ValueError:
        raise ProjectBootstrapError(
            "Se importó una copia antigua de src "
            f"fuera de CODE_ROOT: {resolved_module}"
        ) from None
    return resolved_module
