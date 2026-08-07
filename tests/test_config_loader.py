"""Golden config composition, schema, resolution, and replay contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sati.config_loader import (
    ConfigCompositionError,
    compose_and_resolve,
    packaged_defaults_dir,
    resolve_configuration,
)
from sati.config_schemas import RootConfig, bare_path_fields
from sati.hashing import (
    model_config_hash,
    preprocessing_hash,
    recombination_hash,
    target_schema_hash,
)
from sati.lifecycle import prepare_target, write_config_bundle

REQUIRED = [
    "data.source.uri=datasets/demo/annotations.json",
    "model.num_classes=3",
]


def _resolve(overrides=(), **kwargs):
    return compose_and_resolve(
        [*REQUIRED, *overrides],
        command=kwargs.pop("command", "infer"),
        **kwargs,
    )


def test_packaged_defaults_dir_is_real():
    assert (packaged_defaults_dir() / "config.yaml").is_file()


def test_defaults_compose_and_validate(tmp_path):
    cfg = _resolve(base_dir=tmp_path)
    assert cfg.experiment.name == "starter"
    assert cfg.data.source.format == "coco"
    assert cfg.runtime.command == "infer"
    assert cfg.runtime.config_id
    assert cfg.runtime.run_id != "{coolname:noseed}"


def test_strict_schema_rejects_legacy_and_deferred_keys(tmp_path):
    base = _resolve(base_dir=tmp_path).model_dump(mode="json")
    base["output"] = {"run_dir": "old"}
    with pytest.raises(ValidationError, match="output"):
        RootConfig.model_validate(base)
    base.pop("output")
    base["sweeps"] = {}
    with pytest.raises(ValidationError, match="sweeps"):
        RootConfig.model_validate(base)


def test_unknown_and_out_of_range_keys_rejected(tmp_path):
    with pytest.raises(Exception):
        _resolve(["slicing.bogus_knob=1"], base_dir=tmp_path)
    with pytest.raises(ValidationError):
        _resolve(["slicing.overlap_ratio=1.0"], base_dir=tmp_path)


def test_paths_and_active_output_resolve_without_writes(tmp_path):
    cfg = _resolve(base_dir=tmp_path)
    assert cfg.data.source.uri == tmp_path / "datasets/demo/annotations.json"
    assert cfg.infer_outputs.dir.is_absolute()
    assert cfg.infer_outputs.results.dir == cfg.infer_outputs.dir / "results"
    assert cfg.training_outputs.dir is None
    assert not cfg.infer_outputs.dir.exists()
    assert cfg.model.imgsz == cfg.slicing.nominal_slice_size


def test_structured_model_weight_invariants(tmp_path):
    checkpoint = tmp_path / "init.pt"
    checkpoint.write_bytes(b"init")
    cfg = _resolve(
        [
            "model.weights.source=checkpoint",
            f"model.weights.uri={checkpoint}",
        ],
        base_dir=tmp_path,
    )
    assert cfg.model.weights.uri == str(checkpoint)
    with pytest.raises(ValidationError, match="required"):
        _resolve(
            ["model.weights.source=checkpoint", "model.weights.uri=null"],
            base_dir=tmp_path,
        )
    with pytest.raises(ValidationError, match="must be absent"):
        _resolve(
            ["model.weights.source=none", "model.weights.uri=anything.pt"],
            base_dir=tmp_path,
        )


def test_resolved_yaml_is_authoritative_exact_replay(tmp_path):
    resolution = resolve_configuration(
        [*REQUIRED, f"infer_outputs.dir={tmp_path}/run"],
        command="infer",
        base_dir=tmp_path,
    )
    root = prepare_target(tmp_path / "run")
    resolved_path = write_config_bundle(root, resolution, argv=["sati", "infer"])
    before = resolved_path.read_bytes()
    again = compose_and_resolve(
        command="infer",
        resolved_config=resolved_path,
        base_dir=tmp_path,
    )
    assert again == resolution.config
    assert resolved_path.read_bytes() == before
    resolved_json = json.loads((root / "config" / "resolved.json").read_text())
    assert resolved_json == yaml.safe_load(resolved_path.read_text())
    with pytest.raises(ConfigCompositionError, match="exact replay"):
        compose_and_resolve(
            ["results.cache_slice_predictions=true"],
            command="infer",
            resolved_config=resolved_path,
        )


def test_root_sources_are_exclusive(tmp_path):
    authored = tmp_path / "root.yaml"
    authored.write_text((packaged_defaults_dir() / "config.yaml").read_text())
    with pytest.raises(ConfigCompositionError, match="mutually exclusive"):
        compose_and_resolve(
            ["experiment=example", *REQUIRED],
            command="infer",
            config_file=authored,
        )


def test_external_authored_root_composes_packaged_groups(tmp_path):
    root = tmp_path / "external.yaml"
    root.write_text(
        "# @package _global_\n"
        "defaults:\n"
        "  - /base: default\n"
        "  - /data: coco\n"
        "  - /model: yolo26n\n"
        "  - /slicing: default\n"
        "  - /recombination: default\n"
        "  - /training: default\n"
        "  - _self_\n"
        "experiment: {name: external}\n"
    )
    cfg = compose_and_resolve(
        REQUIRED,
        command="infer",
        config_file=root,
        base_dir=tmp_path,
    )
    assert cfg.experiment.name == "external"


def test_local_configs_shadow_packaged_fragment(tmp_path, monkeypatch):
    local = tmp_path / "configs"
    (local / "slicing").mkdir(parents=True)
    shutil.copy(packaged_defaults_dir() / "config.yaml", local / "config.yaml")
    (local / "slicing" / "default.yaml").write_text(
        "nominal_slice_size: 512\n"
        "overlap_ratio: 0.25\n"
        "max_levels: null\n"
        "min_area_ratio: 0.2\n"
        "size_gate_px: 16\n"
        "image_format: png\n"
        "level_policy: {kind: auto}\n"
    )
    monkeypatch.chdir(tmp_path)
    cfg = _resolve(base_dir=tmp_path)
    assert cfg.slicing.nominal_slice_size == 512
    assert cfg.recombination.within_level_threshold == 0.8


def test_packaged_and_local_experiment_roots(tmp_path, monkeypatch):
    packaged = _resolve(["experiment=example"], base_dir=tmp_path)
    assert packaged.experiment.name == "example"
    exp = tmp_path / "configs" / "experiment"
    exp.mkdir(parents=True)
    (exp / "mine.yaml").write_text(
        "# @package _global_\n"
        "defaults:\n"
        "  - /base: default\n"
        "  - /data: coco\n"
        "  - /model: yolo26n\n"
        "  - /slicing: default\n"
        "  - /recombination: default\n"
        "  - /training: default\n"
        "  - _self_\n"
        "experiment: {name: mine}\n"
        "slicing: {nominal_slice_size: 768}\n"
    )
    monkeypatch.chdir(tmp_path)
    local = _resolve(["experiment=mine"], base_dir=tmp_path)
    assert local.experiment.name == "mine"
    assert local.slicing.nominal_slice_size == 768


def test_experiment_selector_rejects_yaml_path(tmp_path):
    with pytest.raises(ConfigCompositionError, match="use experiment=example"):
        _resolve(["experiment=configs/experiment/example.yaml"], base_dir=tmp_path)


def test_image_root_is_coco_only(tmp_path):
    assert _resolve(["data.source.image_root=imgs"], base_dir=tmp_path).data.source.image_root
    with pytest.raises(ValidationError, match="only supported"):
        _resolve(
            [
                "data=yolo",
                "data.source.uri=x",
                "+data.source.image_root=imgs",
            ],
            base_dir=tmp_path,
        )


def test_reused_prepared_product_derives_prepared_dataset_id(tmp_path):
    from sati.hashing import prepared_dataset_id
    from sati.manifests import (
        SCHEMA_VERSION,
        HashWithSource,
        ManifestMetadata,
        write_metadata_sidecar,
    )

    prepared = tmp_path / "prepared"
    (prepared / "manifests").mkdir(parents=True)
    prepared_hash = "sha256:" + "a" * 64
    write_metadata_sidecar(
        ManifestMetadata(
            schema_version=SCHEMA_VERSION,
            class_mapping={"0": "obj"},
            hashes={
                "dataset_hash": HashWithSource(value="sha256:" + "d" * 64, source={}),
                "prepared_dataset_hash": HashWithSource(value=prepared_hash, source={}),
            },
        ),
        prepared / "manifests",
    )
    cfg = compose_and_resolve(
        [
            "data.source.uri=unused.json",
            "model.num_classes=1",
            f"data.prepared.uri={prepared}",
            f"training_outputs.dir={tmp_path}/run",
        ],
        command="train",
        base_dir=tmp_path,
    )
    assert cfg.data.prepared.prepared_dataset_hash == prepared_hash
    # Regression: the reuse path copied the hashes but left the ID null.
    assert cfg.data.prepared.prepared_dataset_id == prepared_dataset_id(prepared_hash)


def test_inspection_resolution_is_dataset_free(tmp_path):
    args = [
        "data.source.uri=/does/not/exist.json",
        "model.num_classes=1",
        f"training_outputs.dir={tmp_path}/run",
    ]
    # Full resolution ingests the raw dataset and fails when it is absent.
    with pytest.raises(Exception):
        resolve_configuration(args, command="train", base_dir=tmp_path)
    # Inspection resolution must succeed without the dataset: identities and the
    # active output resolve; dataset-derived identities stay deferred (None).
    cfg = resolve_configuration(
        args, command="train", base_dir=tmp_path, for_inspection=True
    ).config
    assert cfg.runtime.config_hash is not None
    assert cfg.runtime.config_id is not None
    assert cfg.training_outputs.dir is not None
    assert cfg.data.prepared.dataset_hash is None
    assert cfg.data.prepared.prepared_dataset_id is None


def test_no_bare_path_fields():
    assert bare_path_fields(RootConfig) == []


def test_renamed_recombination_thresholds_reject_old_names(tmp_path):
    """The IoU-baked names are gone with no alias (D4).

    Renaming rather than aliasing is only safe because ``extra="forbid"`` turns a
    stale authored config into an immediate error instead of a silent default.
    """
    with pytest.raises(Exception):
        _resolve(["recombination.within_level_iou=0.5"], base_dir=tmp_path)
    with pytest.raises(Exception):
        _resolve(["recombination.cross_level_iou=0.5"], base_dir=tmp_path)


def test_recombination_defaults_are_match_first(tmp_path):
    """Packaged defaults: IoS 0.8 class-agnostic, suppression (D12, D16)."""
    cfg = _resolve(base_dir=tmp_path)
    rec = cfg.recombination
    assert rec.method == "nms"
    assert rec.match_metric == "intersection_over_smaller"
    assert (rec.within_level_threshold, rec.cross_level_threshold) == (0.8, 0.8)
    assert rec.max_merge_growth == 8.0
    assert rec.class_agnostic is True


def test_max_merge_growth_cannot_shrink_a_box(tmp_path):
    with pytest.raises(ValidationError):
        _resolve(["recombination.max_merge_growth=0.9"], base_dir=tmp_path)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ARMS = {
    "combo1": "configs/experiment/v1/combo1/default.yaml",
    "combo1b": "configs/experiment/v1/combo1b/default.yaml",
    "combo1b_nmm": "configs/experiment/v1/combo1b/greedy_nmm.yaml",
}


def _arm(name):
    return resolve_configuration(
        command="infer",
        config_file=_REPO_ROOT / _ARMS[name],
        base_dir=_REPO_ROOT,
        for_inspection=True,
    ).config


def test_ab_comparison_arms_are_controlled():
    """The A/B/C only isolates one variable per pair if these hold (D13, D15).

    combo1 -> combo1b     changes matching only.
    combo1b -> greedy_nmm changes the merge method only.
    Every compatibility identity must be equal across all three, or a metric
    difference cannot be attributed to recombination at all.
    """
    arms = {name: _arm(name) for name in _ARMS}

    compat = {
        name: (
            model_config_hash(cfg.model),
            preprocessing_hash(cfg.slicing),
            target_schema_hash(cfg),
        )
        for name, cfg in arms.items()
    }
    assert len(set(compat.values())) == 1, compat

    # same raw dataset and split, so dataset identity cannot diverge either
    assert len({(c.data.source.uri, c.data.source.split_file) for c in arms.values()}) == 1

    # ...while recombination identity moves for every arm, in both scopes
    for scope in ("level_merge", "final_merge"):
        digests = {
            name: recombination_hash(cfg.recombination, scope)
            for name, cfg in arms.items()
        }
        assert len(set(digests.values())) == 3, (scope, digests)

    # the greedy arm differs from combo1b in `method` and nothing else
    base, nmm = arms["combo1b"].recombination, arms["combo1b_nmm"].recombination
    assert base.method == "nms" and nmm.method == "greedy_nmm"
    assert base.model_dump(exclude={"method"}) == nmm.model_dump(exclude={"method"})
