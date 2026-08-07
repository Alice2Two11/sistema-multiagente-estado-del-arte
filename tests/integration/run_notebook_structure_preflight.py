from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "03B_extraccion_cuantitativa_kb_migrado_v16.ipynb"


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(source)
    required = [
        "RUN_REAL_03B",
        "RUN_DETERMINISTIC_FLATTENING_REPAIR",
        "DETERMINISTIC_REPAIR",
        "execute_quantitative_runtime_transaction",
        "transaction_executed",
    ]
    forbidden = [
        "def build_quant_prompt",
        "def normalize_metric_name",
        "def value_found_in_text",
        "OPENAI_KEY_FILE",
        "getpass",
        "Fernet(",
        "ExtractionAgent",
        "04_analisis",
    ]
    missing = [item for item in required if item not in source]
    duplicated = [item for item in forbidden if item in source]
    if missing or duplicated:
        print(json.dumps({"status": "FAILED", "missing": missing, "forbidden": duplicated}))
        return 1
    print(json.dumps({
        "status": "OK",
        "notebook": NOTEBOOK.name,
        "ast_valid": True,
        "thin_shell": True,
        "explicit_modes": True,
        "dataset_normalization_repair_visible": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
