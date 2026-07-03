"""Compose → validate → resolve behavior (DESIGN-DOC-v1 §3)."""


import pytest
from omegaconf.errors import MissingMandatoryValue
from pydantic import ValidationError

from sati.config_loader import compose_and_resolve, packaged_defaults_dir
from sati.config_schemas import RootConfig, bare_path_fields

REQUIRED = [
    "data.source_path=datasets/demo/annotations.json",
    "model.num_classes=3",
]


def _resolve(overrides=(), **kwargs):
    kwargs.setdefault("write", False)
    return compose_and_resolve([*REQUIRED, *overrides], **kwargs)


def test_packaged_defaults_dir_is_real():
    assert (packaged_defaults_dir() / "config.yaml").is_file()


def test_defaults_compose_and_validate(tmp_path):
    cfg = _resolve(base_dir=tmp_path)
    assert cfg.slicing.nominal_slice_size == 1024
    assert cfg.data.source_format == "coco"
    assert cfg.run_id is not None


def test_missing_mandatory_value_raises(tmp_path):
    with pytest.raises(MissingMandatoryValue):
        compose_and_resolve(["model.num_classes=3"], write=False, base_dir=tmp_path)


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(Exception):  # hydra rejects unknown override keys pre-validation
        _resolve(["slicing.bogus_knob=1"], base_dir=tmp_path)


def test_out_of_range_rejected(tmp_path):
    with pytest.raises(ValidationError):
        _resolve(["slicing.overlap_ratio=1.0"], base_dir=tmp_path)


def test_paths_absolutized_and_imgsz_derived(tmp_path):
    cfg = _resolve(base_dir=tmp_path)
    assert cfg.data.source_path == tmp_path / "datasets/demo/annotations.json"
    assert cfg.output.run_dir.is_absolute()
    assert cfg.model.imgsz == cfg.slicing.nominal_slice_size


def test_explicit_imgsz_survives(tmp_path):
    cfg = _resolve(["model.imgsz=640"], base_dir=tmp_path)
    assert cfg.model.imgsz == 640


def test_resolved_yaml_round_trip(tmp_path):
    cfg = _resolve([f"output.run_dir={tmp_path}/run"], base_dir=tmp_path, write=True)
    resolved = tmp_path / "run" / "resolved.yaml"
    assert resolved.is_file()
    again = compose_and_resolve(resolved_config=resolved, write=False, base_dir=tmp_path)
    assert again.run_id != cfg.run_id  # fresh run_id per invocation
    assert again.model_dump(exclude={"run_id"}) == cfg.model_dump(exclude={"run_id"})


def test_resolved_config_applies_only_overrides(tmp_path):
    cfg = _resolve([f"output.run_dir={tmp_path}/run"], base_dir=tmp_path, write=True)
    again = compose_and_resolve(
        ["training.epochs=7"],
        resolved_config=tmp_path / "run" / "resolved.yaml",
        write=False,
        base_dir=tmp_path,
    )
    assert again.training.epochs == 7
    assert again.slicing == cfg.slicing


def test_local_config_defaults_shadows_fragment(tmp_path, monkeypatch):
    # Fragment-level shadowing: a local tree carrying the primary config plus
    # only the group files it overrides wins for those groups; everything else
    # falls through to the packaged tree. (Hydra requires the primary config in
    # the main dir — a local tree without config.yaml cannot shadow anything.)
    import shutil

    local = tmp_path / "config_defaults"
    (local / "slicing").mkdir(parents=True)
    shutil.copy(packaged_defaults_dir() / "config.yaml", local / "config.yaml")
    (local / "slicing" / "default.yaml").write_text(
        "nominal_slice_size: 512\noverlap_ratio: 0.25\n"
    )
    monkeypatch.chdir(tmp_path)
    cfg = _resolve(base_dir=tmp_path)
    assert cfg.slicing.nominal_slice_size == 512
    assert cfg.slicing.overlap_ratio == 0.25
    assert cfg.recombination.within_level_iou == 0.5  # packaged fallthrough


def test_packaged_experiment_as_primary_root(tmp_path):
    # Hybrid scheme: experiment=name makes experiment/name the primary config.
    cfg = _resolve(["experiment=example"], base_dir=tmp_path)
    assert cfg.slicing.overlap_ratio == 0.25
    assert cfg.training.epochs == 10
    assert cfg.slicing.nominal_slice_size == 1024  # group default via experiment defaults


def test_local_experiment_resolves_packaged_groups(tmp_path, monkeypatch):
    # A local experiment file is a root of its own: its dir becomes primary and
    # every group it references falls through to the packaged tree — no local
    # config.yaml needed.
    exp_dir = tmp_path / "config_defaults" / "experiment"
    exp_dir.mkdir(parents=True)
    (exp_dir / "mine.yaml").write_text(
        "# @package _global_\n"
        "defaults:\n"
        "  - /data: coco\n"
        "  - /model: yolo26n\n"
        "  - /slicing: default\n"
        "  - /recombination: default\n"
        "  - /training: default\n"
        "  - _self_\n"
        "results:\n"
        "  cache_slice_predictions: true\n"
        "output:\n"
        "  run_dir: runs/local\n"
        "slicing:\n"
        "  nominal_slice_size: 768\n"
    )
    monkeypatch.chdir(tmp_path)
    cfg = _resolve(["experiment=mine"], base_dir=tmp_path)
    assert cfg.slicing.nominal_slice_size == 768
    assert cfg.results.cache_slice_predictions is True
    assert cfg.recombination.within_level_iou == 0.5  # packaged fallthrough


def test_experiment_selector_rejects_file_paths(tmp_path):
    from sati.config_loader import ConfigCompositionError

    with pytest.raises(ConfigCompositionError, match="experiment=example"):
        _resolve(["experiment=config_defaults/experiment/example.yaml"], base_dir=tmp_path)


def test_image_root_is_coco_only(tmp_path):
    cfg = _resolve(["data.image_root=imgs"], base_dir=tmp_path)
    assert cfg.data.image_root == tmp_path / "imgs"  # SatiPath → absolutized
    with pytest.raises(ValidationError, match="only supported for source_format=coco"):
        _resolve(["data=yolo", "data.source_path=x", "+data.image_root=imgs"], base_dir=tmp_path)


def test_no_bare_path_fields():
    # Settled decision (v1 §4.7): a bare pathlib.Path in a schema is a bug.
    assert bare_path_fields(RootConfig) == []
