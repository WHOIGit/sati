"""Public compose → validate → resolve API."""

from sati.config_loader.compositor import (
    ComposedConfig,
    ConfigCompositionError,
    compose_config,
    default_config_dirs,
    load_resolved,
    packaged_defaults_dir,
)
from sati.config_loader.conductor import (
    ConfigResolution,
    compose_and_resolve,
    resolve_configuration,
)
from sati.config_loader.resolver import (
    ResolutionResult,
    render_template,
    resolve_config,
    validate_resolved_config,
)

__all__ = [
    "ComposedConfig",
    "ConfigCompositionError",
    "ConfigResolution",
    "ResolutionResult",
    "compose_and_resolve",
    "compose_config",
    "default_config_dirs",
    "load_resolved",
    "packaged_defaults_dir",
    "render_template",
    "resolve_config",
    "resolve_configuration",
    "validate_resolved_config",
]
