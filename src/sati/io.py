"""Image read/decode, COCO/YOLO ingestion adapters, split handling (v1 §4, §4.6).

COCO and YOLO exist **only** here (and in the exporters); everything downstream
speaks manifests and half-open ``xyxy``. Ingestion is pure metadata — no pixel
decode except YOLO's header-only size reads.

Split semantics (§4.6): an optional TOML file with ``training = [...]``,
``validation = [...]``, ``holdout = [...]`` lists of frame identifiers
overrides source splits. Entries may be fullframe ids, image filenames, or
image filename stems. A TOML entry missing from the source is an **error**; a
source frame absent from the TOML is dropped with a **warning**.
"""

from __future__ import annotations

import json
import tomllib
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from sati.errors import SatiError

Box = tuple[float, float, float, float]

SPLITS = ("training", "validation", "holdout")

# YOLO data.yaml keys → SATI split names
_YOLO_SPLIT_KEYS = {"train": "training", "val": "validation", "test": "holdout"}

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class IngestError(SatiError):
    """Raised for malformed or inconsistent source datasets."""


@dataclass(frozen=True)
class IngestedFrame:
    fullframe_id: str  # canonicalized to string at ingest
    source_uri: str  # absolute path of the source image
    width: int
    height: int
    color_mode: str  # "RGB" | "grayscale"
    dataset_split: str


@dataclass(frozen=True)
class IngestedAnnotation:
    fullframe_id: str
    ann_id: int
    class_id: int  # contiguous 0-based SATI index (remapped at ingest)
    class_name: str
    bbox_xyxy: Box  # half-open, full-res pixels


@dataclass(frozen=True)
class IngestedDataset:
    frames: tuple[IngestedFrame, ...]
    annotations: tuple[IngestedAnnotation, ...]
    class_map: dict[int, str]  # ordered contiguous index → name


# --- splits (§4.6) ----------------------------------------------------------


def load_split_toml(path: Path | str) -> dict[str, str]:
    """Parse split TOML into ``entry → split``; duplicate entries are errors.

    Entries are resolved to fullframe ids by :func:`apply_split_override`, which
    has access to source image filenames.
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    unknown = set(data) - set(SPLITS)
    if unknown:
        raise IngestError(f"split file {path}: unknown split keys {sorted(unknown)}")
    split_map: dict[str, str] = {}
    for split in SPLITS:
        for frame_id in data.get(split, []):
            frame_id = str(frame_id)
            if frame_id in split_map:
                raise IngestError(
                    f"split file {path}: '{frame_id}' appears in both "
                    f"'{split_map[frame_id]}' and '{split}'"
                )
            split_map[frame_id] = split
    return split_map


def _split_aliases(frame: IngestedFrame) -> tuple[str, ...]:
    source_name = Path(frame.source_uri).name
    source_stem = Path(frame.source_uri).stem
    return tuple(dict.fromkeys((frame.fullframe_id, source_name, source_stem)))


def apply_split_override(
    frames: tuple[IngestedFrame, ...], split_map: dict[str, str]
) -> tuple[IngestedFrame, ...]:
    """§4.6 semantics: TOML entry missing from source = error; source frame
    absent from the TOML = dropped with a warning."""
    aliases: dict[str, set[str]] = {}
    for frame in frames:
        for alias in _split_aliases(frame):
            aliases.setdefault(alias, set()).add(frame.fullframe_id)

    missing = sorted(key for key in split_map if key not in aliases)
    if missing:
        raise IngestError(f"split file entries absent from the source: {missing}")

    ambiguous = sorted(key for key in split_map if len(aliases[key]) > 1)
    if ambiguous:
        raise IngestError(f"split file entries are ambiguous: {ambiguous}")

    resolved: dict[str, str] = {}
    for key, split in split_map.items():
        fullframe_id = next(iter(aliases[key]))
        previous = resolved.get(fullframe_id)
        if previous is not None:
            raise IngestError(
                f"split file assigns frame '{fullframe_id}' more than once "
                f"('{previous}' and '{split}')"
            )
        resolved[fullframe_id] = split

    kept: list[IngestedFrame] = []
    for frame in frames:
        split = resolved.get(frame.fullframe_id)
        if split is None:
            warnings.warn(
                f"frame '{frame.fullframe_id}' absent from split file; dropped",
                stacklevel=2,
            )
            continue
        kept.append(replace(frame, dataset_split=split))
    return tuple(kept)


# --- COCO adapter -----------------------------------------------------------


def _coco_class_map(categories: list[dict]) -> tuple[dict[int, str], dict[int, int]]:
    """COCO category ids (possibly sparse) → contiguous 0-based SATI indices,
    ordered by ascending COCO id."""
    ordered = sorted(categories, key=lambda c: c["id"])
    class_map = {index: cat["name"] for index, cat in enumerate(ordered)}
    remap = {cat["id"]: index for index, cat in enumerate(ordered)}
    return class_map, remap


def _ingest_coco_file(
    path: Path, split: str, image_root: Path | None
) -> tuple[list[IngestedFrame], list[IngestedAnnotation], list[dict]]:
    data = json.loads(path.read_text())
    if image_root is None:
        image_root = path.parent
    frames = [
        IngestedFrame(
            fullframe_id=str(img["id"]),
            source_uri=str((image_root / img["file_name"]).resolve()),
            width=int(img["width"]),
            height=int(img["height"]),
            color_mode="RGB",  # COCO carries no mode; corrected at pixel read
            dataset_split=split,
        )
        for img in data.get("images", [])
    ]
    _, remap = _coco_class_map(data.get("categories", []))
    id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    annotations = []
    for ann in data.get("annotations", []):
        x, y, w, h = ann["bbox"]
        annotations.append(
            IngestedAnnotation(
                fullframe_id=str(ann["image_id"]),
                ann_id=int(ann["id"]),
                class_id=remap[ann["category_id"]],
                class_name=id_to_name[ann["category_id"]],
                bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
            )
        )
    return frames, annotations, data.get("categories", [])


def ingest_coco(
    source_path: Path | str,
    *,
    default_split: str = "training",
    image_root: Path | str | None = None,
) -> IngestedDataset:
    """Ingest COCO: a single annotation JSON, or a directory holding per-split
    files named ``training.json`` / ``validation.json`` / ``holdout.json``.

    COCO ``xywh`` becomes half-open ``xyxy`` here and never appears again.
    ``file_name`` entries resolve against ``image_root`` (``data.image_root``)
    when given, else each annotation file's own directory.
    """
    source_path = Path(source_path)
    if image_root is not None:
        image_root = Path(image_root)
        if not image_root.is_dir():
            raise IngestError(f"data.image_root is not a directory: {image_root}")
    per_split_files: list[tuple[Path, str]]
    if source_path.is_dir():
        per_split_files = [
            (source_path / f"{split}.json", split)
            for split in SPLITS
            if (source_path / f"{split}.json").is_file()
        ]
        if not per_split_files:
            raise IngestError(
                f"no {'{training,validation,holdout}'}.json found in {source_path}"
            )
    elif source_path.is_file():
        per_split_files = [(source_path, default_split)]
    else:
        raise IngestError(f"COCO source not found: {source_path}")

    frames: list[IngestedFrame] = []
    annotations: list[IngestedAnnotation] = []
    all_categories: dict[int, str] = {}
    for path, split in per_split_files:
        f, a, categories = _ingest_coco_file(path, split, image_root)
        for cat in categories:
            existing = all_categories.get(cat["id"])
            if existing is not None and existing != cat["name"]:
                raise IngestError(
                    f"category id {cat['id']} means '{existing}' and '{cat['name']}' "
                    "in different split files"
                )
            all_categories[cat["id"]] = cat["name"]
        frames.extend(f)
        annotations.extend(a)

    seen: dict[str, str] = {}
    for frame in frames:
        if frame.fullframe_id in seen:
            raise IngestError(f"duplicate fullframe_id '{frame.fullframe_id}' across inputs")
        seen[frame.fullframe_id] = frame.dataset_split

    class_map = {
        index: name
        for index, (_, name) in enumerate(sorted(all_categories.items()))
    }
    return IngestedDataset(
        frames=tuple(frames), annotations=tuple(annotations), class_map=class_map
    )


# --- YOLO adapter -----------------------------------------------------------


def _png_size(path: Path) -> tuple[int, int]:
    """Header-only image size read (no full decode)."""
    from PIL import Image

    with Image.open(path) as img:
        return img.size  # (width, height)


def ingest_yolo(source_path: Path | str) -> IngestedDataset:
    """Ingest a YOLO tree via its ``data.yaml`` (path to the file or its dir).

    ``train``/``val``/``test`` keys map to training/validation/holdout.
    Normalized ``cxcywh`` label rows become absolute half-open ``xyxy``.
    Frames are identified by image filename stem (must be dataset-unique).
    """
    import yaml

    source_path = Path(source_path)
    yaml_path = source_path / "data.yaml" if source_path.is_dir() else source_path
    if not yaml_path.is_file():
        raise IngestError(f"data.yaml not found at {yaml_path}")
    spec = yaml.safe_load(yaml_path.read_text())

    names = spec.get("names")
    if isinstance(names, dict):
        class_map = {int(k): str(v) for k, v in sorted(names.items(), key=lambda kv: int(kv[0]))}
    elif isinstance(names, list):
        class_map = dict(enumerate(str(n) for n in names))
    else:
        raise IngestError(f"{yaml_path}: 'names' must be a list or index→name mapping")

    root = Path(spec.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    frames: list[IngestedFrame] = []
    annotations: list[IngestedAnnotation] = []
    ann_id = 0
    for key, split in _YOLO_SPLIT_KEYS.items():
        images_entry = spec.get(key)
        if images_entry is None:
            continue
        images_dir = Path(images_entry)
        if not images_dir.is_absolute():
            images_dir = (root / images_dir).resolve()
        if not images_dir.is_dir():
            raise IngestError(f"{yaml_path}: '{key}' directory not found: {images_dir}")
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            frame_id = image_path.stem
            if any(f.fullframe_id == frame_id for f in frames):
                raise IngestError(f"duplicate fullframe_id '{frame_id}' across YOLO splits")
            width, height = _png_size(image_path)
            frames.append(
                IngestedFrame(
                    fullframe_id=frame_id,
                    source_uri=str(image_path.resolve()),
                    width=width,
                    height=height,
                    color_mode="RGB",
                    dataset_split=split,
                )
            )
            label_path = Path(
                str(image_path.parent).replace("images", "labels", 1)
            ) / f"{frame_id}.txt"
            if not label_path.is_file():
                continue  # negative frame: no annotations
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if class_id not in class_map:
                    raise IngestError(f"{label_path}: class id {class_id} not in names")
                cx, cy, w, h = (float(v) for v in parts[1:5])
                annotations.append(
                    IngestedAnnotation(
                        fullframe_id=frame_id,
                        ann_id=ann_id,
                        class_id=class_id,
                        class_name=class_map[class_id],
                        bbox_xyxy=(
                            (cx - w / 2) * width,
                            (cy - h / 2) * height,
                            (cx + w / 2) * width,
                            (cy + h / 2) * height,
                        ),
                    )
                )
                ann_id += 1
    return IngestedDataset(
        frames=tuple(frames), annotations=tuple(annotations), class_map=class_map
    )


# --- pixels -----------------------------------------------------------------


def read_image(path: Path | str) -> np.ndarray:
    """Decode to RGB ``uint8`` HWC. Grayscale sources come back 3-channel."""
    data = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data is None:
        raise IngestError(f"failed to decode image: {path}")
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)


def detect_color_mode(image: np.ndarray) -> str:
    """"grayscale" iff all channels are identical."""
    if image.ndim == 2:
        return "grayscale"
    if (image[..., 0] == image[..., 1]).all() and (image[..., 1] == image[..., 2]).all():
        return "grayscale"
    return "RGB"
