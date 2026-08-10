# ============================================================
# UTILIDADES PARA LLM
# Credencial de runtime sin usar el almacén de secretos de Colab.
# ============================================================

import os
import re
import json
from pathlib import Path
from getpass import getpass

from langchain_openai import ChatOpenAI


PROJECT_DIR = Path(
    "/content/proyecto_estado_arte"
)

RUNTIME_SECRET_DIR = (
    PROJECT_DIR
    / ".runtime_secrets"
)

OPENAI_KEY_FILE = (
    RUNTIME_SECRET_DIR
    / "openai_api_key.txt"
)


def _normalize_key(value):
    """
    Convierte la credencial a texto limpio sin imprimirla.
    """
    if value is None:
        return ""

    return str(value).strip()


def save_openai_key_for_runtime(api_key):
    """
    Guarda OPENAI_API_KEY únicamente dentro del runtime actual.

    No consulta el almacén de secretos de Colab, por lo que no
    activa el cuadro de confirmación de acceso a secretos.
    """
    api_key = _normalize_key(
        api_key
    )

    if not api_key:
        raise ValueError(
            "No se puede guardar una API key vacía."
        )

    RUNTIME_SECRET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OPENAI_KEY_FILE.write_text(
        api_key,
        encoding="utf-8",
    )

    try:
        OPENAI_KEY_FILE.chmod(
            0o600
        )
    except Exception:
        pass

    os.environ[
        "OPENAI_API_KEY"
    ] = api_key

    return OPENAI_KEY_FILE


def load_openai_key_for_runtime():
    """
    Recupera la clave desde:
    1. la variable de entorno;
    2. el archivo local del runtime.
    """
    api_key = _normalize_key(
        os.environ.get(
            "OPENAI_API_KEY",
            "",
        )
    )

    if api_key:
        return api_key

    if OPENAI_KEY_FILE.exists():
        api_key = _normalize_key(
            OPENAI_KEY_FILE.read_text(
                encoding="utf-8",
            )
        )

        if api_key:
            os.environ[
                "OPENAI_API_KEY"
            ] = api_key

    return api_key


def ensure_openai_key(
    allow_prompt=True,
    persist_if_prompted=True,
):
    """
    Asegura que OPENAI_API_KEY esté disponible.

    La clave se solicita únicamente cuando:
    - no existe en os.environ; y
    - no existe en el archivo local del runtime.

    Después se reutiliza automáticamente durante la sesión.
    """
    api_key = (
        load_openai_key_for_runtime()
    )

    prompted = False

    if not api_key and allow_prompt:
        api_key = _normalize_key(
            getpass(
                "Pega tu OPENAI_API_KEY "
                "una sola vez para este runtime: "
            )
        )
        prompted = True

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY no está disponible. "
            "Ejecuta primero el notebook 00."
        )

    os.environ[
        "OPENAI_API_KEY"
    ] = api_key

    if (
        persist_if_prompted
        or prompted
    ):
        save_openai_key_for_runtime(
            api_key
        )

    return api_key


def get_openai_api_key():
    """
    Devuelve la clave cargada por el notebook 00.
    """
    return ensure_openai_key(
        allow_prompt=False,
        persist_if_prompted=False,
    )


def get_llm(
    model=None,
    temperature=0.0,
    **kwargs,
):
    """
    Crea ChatOpenAI usando la credencial local del runtime.
    """
    if model is None:
        try:
            from config import OPENAI_MODEL
            model = OPENAI_MODEL
        except Exception:
            model = "gpt-4.1-mini"

    return ChatOpenAI(
        model=model,
        temperature=float(
            temperature
        ),
        api_key=get_openai_api_key(),
        **kwargs,
    )


def parse_json_safely(text):
    """
    Extrae JSON incluso cuando el modelo lo envuelve
    en un bloque Markdown.
    """
    if isinstance(
        text,
        (dict, list),
    ):
        return text

    value = str(text).strip()

    value = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\s*```\s*$",
        "",
        value,
    )
    value = value.strip()

    try:
        return json.loads(
            value
        )
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for index, character in enumerate(
        value
    ):
        if character not in "{[":
            continue

        try:
            parsed, _ = (
                decoder.raw_decode(
                    value[index:]
                )
            )
            return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError(
        "La respuesta del LLM no contiene JSON válido."
    )
