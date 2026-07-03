"""Parquet manifest schemas, column-by-column writers, sidecar (v1 §4).

Every schema is **explicit and hand-written** — Arrow type inference is
forbidden (it breaks on null-leading and list columns). Tables are built
column-by-column against the declared schema: a row-dict that omits a column
yields a typed null; unknown keys are dropped.

The ``_metadata.json`` sidecar pairs each compatibility hash with the exact
``*_source`` object it was computed from, so every hash is independently
re-verifiable. Its writer must **not** sort keys — numeric-string class-map
keys ("0", "1", "10") keep insertion order — which is the opposite of the
hash serializer's setting, so the two never share a JSON-dump helper (§4.5).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from sati.hashing import sha256_of, canonical_json

SCHEMA_VERSION = 1

METADATA_SIDECAR_NAME = "_metadata.json"

_INT2 = pa.list_(pa.int32(), 2)
_F4 = pa.list_(pa.float64(), 4)
_STR_LIST = pa.list_(pa.string())

FULLFRAMES_SCHEMA = pa.schema(
    [
        pa.field("fullframe_id", pa.string(), nullable=False),
        pa.field("source_uri", pa.string(), nullable=False),
        pa.field("dataset_split", pa.string(), nullable=False),
        pa.field("fullres_wh", _INT2, nullable=False),
        pa.field("color_mode", pa.string(), nullable=False),
    ]
)

ANNOTATIONS_SCHEMA = pa.schema(
    [
        pa.field("fullframe_id", pa.string(), nullable=False),
        pa.field("ann_id", pa.int64(), nullable=False),
        pa.field("class_id", pa.int32(), nullable=False),
        pa.field("class_name", pa.string(), nullable=False),
        pa.field("bbox_xyxy", _F4, nullable=False),
        pa.field("bbox_w", pa.float64(), nullable=False),
        pa.field("bbox_h", pa.float64(), nullable=False),
        pa.field("bbox_rel_cxcywh", _F4, nullable=False),
    ]
)

SLICES_SCHEMA = pa.schema(
    [
        pa.field("fullframe_id", pa.string(), nullable=False),
        pa.field("level", pa.int32(), nullable=False),
        pa.field("slice_id", pa.int64(), nullable=False),
        pa.field("dataset_split", pa.string(), nullable=False),
        pa.field("scale", pa.float64(), nullable=False),
        pa.field("level_wh", _INT2, nullable=False),
        pa.field("slice_xy", _INT2, nullable=False),
        pa.field("slice_wh", _INT2, nullable=False),
        pa.field("emitted_wh", _INT2, nullable=False),
        pa.field("is_whole_level", pa.bool_(), nullable=False),
        pa.field("stretched_axes", _STR_LIST, nullable=False),
        pa.field("slice_uri", pa.string(), nullable=False),
        pa.field("affine_slice_to_level", _F4, nullable=False),
        pa.field("affine_slice_to_fullres", _F4, nullable=False),
    ]
)

SLICE_ANNOTATIONS_SCHEMA = pa.schema(
    [
        pa.field("slice_ann_id", pa.int64(), nullable=False),
        pa.field("parent_ann_id", pa.int64(), nullable=False),
        pa.field("fullframe_id", pa.string(), nullable=False),
        pa.field("level", pa.int32(), nullable=False),
        pa.field("slice_id", pa.int64(), nullable=False),
        pa.field("bbox_fullres_xyxy", _F4, nullable=False),
        pa.field("bbox_level_xyxy", _F4, nullable=False),
        pa.field("bbox_slice_xyxy", _F4, nullable=False),
        pa.field("visible_area_frac", pa.float64(), nullable=False),
        pa.field("truncated_edges", _STR_LIST, nullable=False),
        pa.field("dropped_by_size_gate", pa.bool_(), nullable=False),
        pa.field("dropped_by_min_area_ratio", pa.bool_(), nullable=False),
    ]
)

# One tall table discriminated by record_type (§4.5). Stage-specific columns
# are nullable; omission in a row-dict yields a typed null. Nullable geometry
# columns use variable-size lists — parquet cannot write null entries in
# fixed-size lists ("Lists with non-zero length null components").
VALIDATION_RESULTS_SCHEMA = pa.schema(
    [
        pa.field("record_type", pa.string(), nullable=False),
        pa.field("fullframe_id", pa.string(), nullable=False),
        pa.field("dataset_split", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("schema_version", pa.int32(), nullable=False),
        # detection payload (slice_prediction / level_merge / final_merge)
        pa.field("level", pa.int32()),
        pa.field("slice_id", pa.int64()),
        pa.field("bbox_slice_xyxy", pa.list_(pa.float64())),
        pa.field("bbox_level_xyxy", pa.list_(pa.float64())),
        pa.field("bbox_fullres_xyxy", pa.list_(pa.float64())),
        pa.field("score", pa.float64()),
        pa.field("class_id", pa.int32()),
        pa.field("source_slice_ids", pa.list_(pa.int64())),
        pa.field("source_levels", pa.list_(pa.int32())),
        pa.field("merge_strategy_name", pa.string()),
        pa.field("recombination_hash", pa.string()),
        # frame_metadata payload
        pa.field("source_uri", pa.string()),
        pa.field("fullres_wh", pa.list_(pa.int32())),
        # hash family (§4.7), constant per run but carried per row for group-bys
        pa.field("config_hash", pa.string()),
        pa.field("model_config_hash", pa.string()),
        pa.field("checkpoint_hash", pa.string()),
        pa.field("dataset_hash", pa.string()),
        pa.field("class_map_hash", pa.string()),
        pa.field("preprocessing_hash", pa.string()),
    ]
)

RECORD_TYPES = ("slice_prediction", "level_merge", "final_merge", "frame_metadata")


def build_table(schema: pa.Schema, rows: Iterable[dict[str, Any]]) -> pa.Table:
    """Column-by-column construction against an explicit schema (§4.5).

    Missing keys become typed nulls; unknown keys are dropped. Never uses
    ``pa.Table.from_pylist`` type inference.
    """
    row_list = list(rows)
    columns = [
        pa.array([row.get(field.name) for row in row_list], type=field.type)
        for field in schema
    ]
    return pa.Table.from_arrays(columns, schema=schema)


def write_manifest(table: pa.Table, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def read_manifest(path: Path | str, schema: pa.Schema) -> pa.Table:
    """Read a manifest and assert it matches the declared schema exactly."""
    table = pq.read_table(path)
    if table.schema != schema:
        raise ValueError(
            f"manifest schema mismatch at {path}:\n"
            f"expected: {schema}\nfound: {table.schema}"
        )
    return table


class HashWithSource(BaseModel):
    """A compatibility hash paired with the exact source object it digests."""

    model_config = ConfigDict(extra="forbid")

    value: str
    source: Any


class ManifestMetadata(BaseModel):
    """Strict model for the ``_metadata.json`` sidecar (§4.5)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    class_map: dict[str, str]  # index-as-string → name; insertion order is meaning
    hashes: dict[str, HashWithSource]
    column_descriptions: dict[str, str] = {}


def write_metadata_sidecar(metadata: ManifestMetadata, directory: Path | str) -> Path:
    """Write the sidecar. Never ``sort_keys`` — class-map order is meaningful."""
    path = Path(directory) / METADATA_SIDECAR_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=False))
    return path


def read_metadata_sidecar(directory: Path | str) -> ManifestMetadata:
    path = Path(directory) / METADATA_SIDECAR_NAME
    return ManifestMetadata.model_validate_json(path.read_text())


def verify_metadata_hashes(metadata: ManifestMetadata) -> dict[str, bool]:
    """Recompute every hash from its paired source; True where it matches."""
    return {
        name: sha256_of(canonical_json(entry.source)) == entry.value
        for name, entry in metadata.hashes.items()
    }
