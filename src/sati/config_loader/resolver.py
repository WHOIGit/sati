"""Authored → resolved: run_id, path absolutization, derived values (v1 §3.1).

Resolution is type-driven: path fields are discovered by their ``SatiPath``
annotation, never by name, so the resolver and the config hasher can never
disagree about what is a path.
"""

from __future__ import annotations

import copy
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from sati.config_schemas import RootConfig, SatiPathMarker, annotated_field_paths

RESOLVED_CONFIG_NAME = "resolved.yaml"


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def _get_nested(cfg: RootConfig, path: tuple[str, ...]) -> Any:
    node: Any = cfg
    for key in path:
        node = getattr(node, key)
    return node


def _set_nested(cfg: RootConfig, path: tuple[str, ...], value: Any) -> None:
    node: Any = cfg
    for key in path[:-1]:
        node = getattr(node, key)
    setattr(node, path[-1], value)


def resolve_config(
    container: dict[str, Any],
    *,
    base_dir: Path | None = None,
    run_id: str | None = None,
) -> RootConfig:
    """Validate a composed container and materialize the resolved form.

    Absolutizes every ``SatiPath`` field against ``base_dir`` (default CWD) and
    fills derived values (``model.imgsz`` defaults to ``slicing.nominal_slice_size``).
    """
    container = copy.deepcopy(container)
    container["run_id"] = run_id or new_run_id()
    cfg = RootConfig.model_validate(container)

    base = (base_dir or Path.cwd()).resolve()
    for path in annotated_field_paths(RootConfig, SatiPathMarker):
        value = _get_nested(cfg, path)
        if value is None:
            continue
        value = Path(value)
        if not value.is_absolute():
            _set_nested(cfg, path, base / value)

    if cfg.model.imgsz is None:
        cfg.model.imgsz = cfg.slicing.nominal_slice_size
    return cfg


def write_resolved(cfg: RootConfig) -> Path:
    """Write ``resolved.yaml`` into the run directory; returns its path."""
    run_dir = cfg.output.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / RESOLVED_CONFIG_NAME
    out.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    return out
