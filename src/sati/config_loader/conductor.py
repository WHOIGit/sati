"""The config-loader conductor: compose → validate → resolve in one call (v1 §3.1).

The single entry point every caller (CLI, tests, orchestrators) uses so none of
them drift on composition or validation order.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf

from sati.config_loader.compositor import compose_config, load_resolved
from sati.config_loader.resolver import resolve_config, write_resolved
from sati.config_schemas import RootConfig


def compose_and_resolve(
    overrides: Sequence[str] = (),
    *,
    config_dirs: Sequence[Path | str] = (),
    resolved_config: Path | str | None = None,
    base_dir: Path | None = None,
    run_id: str | None = None,
    write: bool = True,
) -> RootConfig:
    """Compose (or reload a resolved artifact), validate, resolve, persist.

    ``write=False`` is for read-only consumers (``sati inspect config``, tests)
    that must not create run directories. A ``--resolved-config`` source keeps
    its values verbatim but gets a fresh ``run_id`` (config_hash is unaffected —
    run_id is derived-only).
    """
    if resolved_config is not None:
        cfg = load_resolved(resolved_config, overrides)
    else:
        cfg = compose_config(overrides, config_dirs=[Path(p) for p in config_dirs]).config

    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(container, dict):
        raise ValueError("composed config root must be a mapping")
    container.pop("run_id", None)

    resolved = resolve_config(container, base_dir=base_dir, run_id=run_id)
    if write:
        write_resolved(resolved)
    return resolved
