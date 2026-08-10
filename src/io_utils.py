# ============================================================
# UTILIDADES DE ENTRADA/SALIDA Y CONTROL DE EXPERIMENTOS
# ============================================================

import json
import re
import pandas as pd
from pathlib import Path


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_if_exists(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return load_json(path)


def save_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path):
    path = Path(path)
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_csv(path):
    return pd.read_csv(path)


def list_experiments(project_dir):
    project_dir = Path(project_dir)
    pattern = re.compile(r"^experimento_paper_(\d+)$")
    experiments = []

    if not project_dir.exists():
        return experiments

    for item in project_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                experiments.append({
                    "number": int(match.group(1)),
                    "experiment_id": item.name,
                    "path": str(item),
                })

    return sorted(experiments, key=lambda x: x["number"])


def load_active_experiment(project_dir):
    project_dir = Path(project_dir)
    return load_json_if_exists(project_dir / "active_experiment.json", default=None)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
