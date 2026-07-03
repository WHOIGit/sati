"""Compose → validate → resolve via the Hydra Compose API (DESIGN-DOC-v1 §3).

- ``compositor`` — Hydra Compose API, searchpath fallthrough, ``--resolved-config``.
- ``resolver`` — run_id, ``SatiPath`` absolutization, derived values, ``resolved.yaml``.
- ``conductor`` — sequences the above into :func:`compose_and_resolve`.
"""

from sati.config_loader.compositor import (
    ComposedConfig,
    ConfigCompositionError,
    compose_config,
    default_config_dirs,
    load_resolved,
    packaged_defaults_dir,
)
from sati.config_loader.conductor import compose_and_resolve
from sati.config_loader.resolver import (
    RESOLVED_CONFIG_NAME,
    new_run_id,
    resolve_config,
    write_resolved,
)

__all__ = [
    "RESOLVED_CONFIG_NAME",
    "ComposedConfig",
    "ConfigCompositionError",
    "compose_and_resolve",
    "compose_config",
    "default_config_dirs",
    "load_resolved",
    "new_run_id",
    "packaged_defaults_dir",
    "resolve_config",
    "write_resolved",
]
