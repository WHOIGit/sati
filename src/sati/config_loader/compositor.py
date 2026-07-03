"""Hydra composition wrapper for SATI configs (DESIGN-DOC-v1 §3.2).

Drives Hydra through the Compose API only — never ``@hydra.main``, never
multirun. Mirrors dojo's compositor: explicit ``--config-dir`` dirs, then a
local ``./config_defaults``, then the packaged tree, joined by
``hydra.searchpath`` so shadowing is fragment-level: a shadowing tree carries
the primary ``config.yaml`` plus only the group fragments it overrides;
everything it omits falls through to the packaged tree. (Hydra constraint: the
primary config must live in the main dir, and group lookups search the main
dir before the searchpath — so a local tree without ``config.yaml`` shadows
nothing.)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


class ConfigCompositionError(RuntimeError):
    """Raised when a config root cannot be selected or composed."""


@dataclass(frozen=True)
class ComposedConfig:
    config: DictConfig
    config_name: str
    config_dirs: tuple[Path, ...]
    overrides: tuple[str, ...]


def packaged_defaults_dir() -> Path:
    """Concrete directory of the packaged Hydra defaults.

    Settled decision (v1 §3.2): resolve ``importlib.resources`` to a real
    filesystem directory for ``initialize_config_dir``; zipped installs are
    unsupported and fail hard here. No ``SearchPathPlugin``.
    """
    traversable = files("sati") / "config_defaults"
    path = Path(str(traversable))
    if not path.is_dir():
        raise ConfigCompositionError(
            "Packaged config defaults are not a real directory "
            f"({traversable!r}); SATI does not support zip-imported installs."
        )
    return path


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def default_config_dirs(extra_config_dirs: Iterable[Path] = ()) -> list[Path]:
    """Explicit dirs, then local ``./config_defaults``, then packaged defaults."""
    candidates = [Path(p) for p in extra_config_dirs]
    for candidate in candidates:
        if not candidate.is_dir():
            raise ConfigCompositionError(
                f"--config-dir does not exist or is not a directory: {candidate}"
            )
    local = Path.cwd() / "config_defaults"
    if local.is_dir():
        candidates.append(local)
    candidates.append(packaged_defaults_dir())
    return _dedupe_paths(candidates)


def _split_experiment_selector(overrides: list[str]) -> tuple[str | None, list[str]]:
    """Intercept a dojo-style ``experiment=name`` selector.

    ``experiment=foo`` makes ``experiment/foo`` the *primary* config (a full
    root with its own defaults list), bypassing the packaged ``config.yaml``.
    Without a selector, ``config.yaml`` stays the root — the hybrid scheme.
    """
    remaining: list[str] = []
    experiment_name: str | None = None
    for override in overrides:
        if experiment_name is None and override.startswith("experiment="):
            experiment_name = override.split("=", 1)[1]
            continue
        remaining.append(override)
    if experiment_name is None:
        return None, remaining
    if not experiment_name:
        raise ConfigCompositionError("experiment= selector cannot be empty")
    if experiment_name.endswith((".yaml", ".yml")) or "config_defaults/" in experiment_name:
        stripped = experiment_name.rsplit("experiment/", 1)[-1]
        stripped = stripped.removesuffix(".yaml").removesuffix(".yml")
        raise ConfigCompositionError(
            "experiment= selects a config group entry, not a YAML file path. "
            f"Use experiment={stripped} instead."
        )
    return f"experiment/{experiment_name}", remaining


def _owning_dir_first(config_name: str, config_dirs: list[Path]) -> list[Path]:
    """Reorder so the dir containing ``config_name`` is primary.

    Hydra cannot load a *primary* config that lives only on the runtime
    ``hydra.searchpath``; the primary config must be in the dir passed to
    ``initialize_config_dir``. The remaining dirs stay on the searchpath so
    group defaults still fall through.
    """
    candidates = [f"{config_name}.yaml", f"{config_name}.yml"]
    for index, config_dir in enumerate(config_dirs):
        if any((config_dir / c).exists() for c in candidates):
            return [config_dir, *(d for i, d in enumerate(config_dirs) if i != index)]
    return list(config_dirs)


def compose_config(
    overrides: Iterable[str] = (),
    *,
    config_dirs: Iterable[Path] = (),
    config_name: str | None = None,
) -> ComposedConfig:
    """Compose the authored config. All Hydra usage ends inside this function.

    Root selection (hybrid scheme): an ``experiment=name`` override makes
    ``experiment/name`` the primary config; otherwise the packaged/local
    ``config.yaml`` is the root. The dir owning the primary config becomes the
    main dir, so a local experiment gets packaged groups via fallthrough.
    """
    all_overrides = list(overrides)
    override_list = all_overrides
    if config_name is None:
        config_name, override_list = _split_experiment_selector(all_overrides)
        if config_name is None:
            config_name = "config"
    search_dirs = _owning_dir_first(config_name, default_config_dirs(config_dirs))
    primary, fallback = search_dirs[0], search_dirs[1:]

    hydra_overrides = list(override_list)
    if fallback:
        searchpath = ",".join(p.resolve().as_uri() for p in fallback)
        hydra_overrides.append(f"hydra.searchpath=[{searchpath}]")

    with initialize_config_dir(config_dir=str(primary), version_base=None):
        cfg = compose(config_name=config_name, overrides=hydra_overrides)
    return ComposedConfig(
        config=cfg,
        config_name=config_name,
        config_dirs=tuple(search_dirs),
        overrides=tuple(all_overrides),
    )


def load_resolved(
    resolved_path: Path | str,
    overrides: Iterable[str] = (),
) -> DictConfig:
    """Load a previously resolved artifact, skip composition, apply only overrides.

    Backs the CLI ``--resolved-config`` path (v1 §13).
    """
    path = Path(resolved_path)
    if not path.is_file():
        raise ConfigCompositionError(f"resolved config not found: {path}")
    cfg = OmegaConf.load(path)
    override_list = list(overrides)
    if override_list:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(override_list))
    assert isinstance(cfg, DictConfig)
    return cfg
