from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = config_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        data = json.loads(raw)
    else:
        data = yaml.safe_load(raw)

    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {config_path} must deserialize into a mapping.")
    return data


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
