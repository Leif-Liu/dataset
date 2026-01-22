from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def resolve_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    cfg = dict(config)
    system_cfg = dict(cfg.get("system", {}))
    if "requirements_dir" in system_cfg:
        system_cfg["requirements_dir"] = resolve_path(
            base_dir, system_cfg["requirements_dir"]
        )
    if "output_dir" in system_cfg:
        system_cfg["output_dir"] = resolve_path(base_dir, system_cfg["output_dir"])
    cfg["system"] = system_cfg
    rag_cfg = dict(cfg.get("rag", {}))
    if "persist_dir" in rag_cfg:
        rag_cfg["persist_dir"] = resolve_path(base_dir, rag_cfg["persist_dir"])
    cfg["rag"] = rag_cfg

    kg_cfg = dict(cfg.get("kg", {}))
    if "persist_dir" in kg_cfg:
        kg_cfg["persist_dir"] = resolve_path(base_dir, kg_cfg["persist_dir"])
    cfg["kg"] = kg_cfg
    return cfg

