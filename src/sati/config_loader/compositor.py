"""Hydra composition and root-source selection for SATI."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import Iterable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from sati.errors import SatiError


class ConfigCompositionError(SatiError):
    """Raised when config sources or Hydra composition are invalid."""


@dataclass(frozen=True)
class ComposedConfig:
    config: DictConfig
    config_name: str
    config_dirs: tuple[Path, ...]
    overrides: tuple[str, ...]
    resolved_source: bool = False


def packaged_defaults_dir() -> Path:
    traversable = files("sati") / "config_defaults"
    path = Path(str(traversable))
    if not path.is_dir():
        raise ConfigCompositionError(
            "packaged config defaults are not a real directory; "
            "SATI does not support zip-imported installs"
        )
    return path


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def default_config_dirs(extra_config_dirs: Iterable[Path] = ()) -> list[Path]:
    """Explicit directories, then automatic ``./configs``, then package defaults."""
    candidates = [Path(path) for path in extra_config_dirs]
    for candidate in candidates:
        if not candidate.is_dir():
            raise ConfigCompositionError(
                f"--config-dir does not exist or is not a directory: {candidate}"
            )
    local = Path.cwd() / "configs"
    if local.is_dir():
        candidates.append(local)
    candidates.append(packaged_defaults_dir())
    return _dedupe_paths(candidates)


def _split_experiment_selector(overrides: list[str]) -> tuple[str | None, list[str]]:
    selectors = [token for token in overrides if token.startswith("experiment=")]
    if len(selectors) > 1:
        raise ConfigCompositionError("only one experiment=<name> selector is allowed")
    remaining = [token for token in overrides if not token.startswith("experiment=")]
    if not selectors:
        return None, remaining
    name = selectors[0].split("=", 1)[1]
    if not name:
        raise ConfigCompositionError("experiment= selector cannot be empty")
    if name.endswith((".yaml", ".yml")) or name.startswith(("./configs/", "configs/")):
        hint = name.removeprefix("./configs/").removeprefix("configs/")
        hint = hint.removeprefix("experiment/").removesuffix(".yaml").removesuffix(".yml")
        raise ConfigCompositionError(
            "experiment= selects a Hydra group entry, not a YAML path; "
            f"use experiment={hint} or --config {name}"
        )
    return f"experiment/{name}", remaining


def _owning_dir_first(config_name: str, dirs: list[Path]) -> list[Path]:
    candidates = (f"{config_name}.yaml", f"{config_name}.yml")
    for index, directory in enumerate(dirs):
        if any((directory / candidate).is_file() for candidate in candidates):
            return [directory, *(item for i, item in enumerate(dirs) if i != index)]
    return dirs


def _relative_name(config_file: Path, dirs: Iterable[Path]) -> tuple[str, Path] | None:
    config_file = config_file.resolve()
    for directory in dirs:
        try:
            relative = config_file.relative_to(directory.resolve())
        except ValueError:
            continue
        if relative.suffix in {".yaml", ".yml"}:
            return relative.with_suffix("").as_posix(), directory.resolve()
    return None


def _compose_search_path(
    config_name: str,
    dirs: list[Path],
    overrides: list[str],
) -> ComposedConfig:
    dirs = _owning_dir_first(config_name, dirs)
    if not dirs:
        raise ConfigCompositionError("no config directories are available")
    primary, fallback = dirs[0], dirs[1:]
    yaml_path = primary / f"{config_name}.yaml"
    yml_path = primary / f"{config_name}.yml"
    alias_context = (
        TemporaryDirectory()
        if yml_path.is_file() and not yaml_path.is_file()
        else nullcontext(None)
    )
    with alias_context as alias_dir:
        composition_primary = primary
        fallback_dirs = fallback
        if alias_dir is not None:
            composition_primary = Path(alias_dir)
            alias_path = composition_primary / f"{config_name}.yaml"
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            copyfile(yml_path, alias_path)
            fallback_dirs = dirs
        hydra_overrides = list(overrides)
        if fallback_dirs:
            uris = ",".join(path.resolve().as_uri() for path in fallback_dirs)
            hydra_overrides.append(f"hydra.searchpath=[{uris}]")
        with initialize_config_dir(
            config_dir=str(composition_primary),
            version_base=None,
        ):
            cfg = compose(config_name=config_name, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    return ComposedConfig(
        config=cfg,
        config_name=config_name,
        config_dirs=tuple(dirs),
        overrides=tuple(overrides),
    )


def compose_config(
    overrides: Iterable[str] = (),
    *,
    config_dirs: Iterable[Path] = (),
    config_file: Path | None = None,
    resolved_config_file: Path | None = None,
) -> ComposedConfig:
    override_list = list(overrides)
    experiment_name, remaining_overrides = _split_experiment_selector(override_list)
    selected = sum(
        source is not None
        for source in (experiment_name, config_file, resolved_config_file)
    )
    if selected > 1:
        raise ConfigCompositionError(
            "experiment=, --config, and --resolved-config are mutually exclusive"
        )

    search_dirs = default_config_dirs(config_dirs)
    if resolved_config_file is not None:
        if remaining_overrides:
            raise ConfigCompositionError(
                "--resolved-config is exact replay and rejects Hydra overrides; "
                "use the run's config/composed.yaml with --config to branch"
            )
        path = resolved_config_file.resolve()
        if not path.is_file():
            raise ConfigCompositionError(f"resolved config does not exist: {path}")
        cfg = OmegaConf.load(path)
        OmegaConf.resolve(cfg)
        return ComposedConfig(
            config=cfg,
            config_name=str(path),
            config_dirs=tuple(search_dirs),
            overrides=(),
            resolved_source=True,
        )

    if config_file is not None:
        path = config_file.resolve()
        if not path.is_file():
            raise ConfigCompositionError(f"config file does not exist: {path}")
        relative = _relative_name(path, search_dirs)
        if relative is None:
            config_name = path.with_suffix("").name
            dirs = _dedupe_paths([path.parent, *search_dirs])
        else:
            config_name, owning_dir = relative
            dirs = _dedupe_paths([owning_dir, *search_dirs])
        return _compose_search_path(config_name, dirs, remaining_overrides)

    config_name = experiment_name
    if config_name is None:
        for directory in search_dirs:
            if any((directory / f"config{suffix}").is_file() for suffix in (".yaml", ".yml")):
                config_name = "config"
                break
    if config_name is None:
        raise ConfigCompositionError(
            "select a root with experiment=<name> or --config PATH"
        )
    return _compose_search_path(config_name, search_dirs, remaining_overrides)


def load_resolved(resolved_path: Path | str) -> DictConfig:
    return compose_config(resolved_config_file=Path(resolved_path)).config
