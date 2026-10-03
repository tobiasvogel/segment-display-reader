from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    SUPPORTED_EXTENSIONS,
    DEFAULT_SEGMENT_TYPE,
    DISPLAY_TYPE_BY_ID,
    SEGMENT_TYPES,
)
from .labels import TYPE_TO_INDEX, padded_target, validate_label


@dataclass(frozen=True)
class SampleMeta:
    path: Path
    display_id: str
    label: str
    index: int
    segment_type: int

    @property
    def source_id(self):
        return (
            f"s{self.segment_type}:{self.display_id}:"
            f"{self.label}:{self.index}"
        )


_FILENAME_RE = re.compile(
    r"^(?:s(?P<segment_type>7|13|14|16)_)?"
    r"d(?P<display>[A-Za-z0-9-]+)_"
    r"(?P<label>[A-Za-z0-9-])_"
    r"(?P<index>\d+)"
    r"(?:_aug\d+)?"
    r"\.[^.]+$"
)


def parse_filename(path):
    path = Path(path)
    match = _FILENAME_RE.match(path.name)
    if not match:
        return None

    display_id = f"d{match.group('display')}"
    explicit_type = match.group("segment_type")

    if explicit_type is not None:
        segment_type = int(explicit_type)
    else:
        segment_type = int(
            DISPLAY_TYPE_BY_ID.get(display_id, DEFAULT_SEGMENT_TYPE)
        )

    if segment_type not in SEGMENT_TYPES:
        return None

    return SampleMeta(
        path=path,
        display_id=display_id,
        label=match.group("label"),
        index=int(match.group("index")),
        segment_type=segment_type,
    )


def collect_samples(*directories):
    samples = []

    for directory in directories:
        directory = Path(directory)
        if not directory.exists():
            continue

        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            meta = parse_filename(path)
            if meta is None:
                continue

            # Fail early on unsupported labels instead of silently training a
            # type-only sample by accident.
            try:
                validate_label(meta.segment_type, meta.label)
            except ValueError as exc:
                raise ValueError(f"{path.name}: {exc}") from exc

            samples.append(meta)

    return samples


class SegmentDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        meta = self.samples[index]

        image = cv2.imread(str(meta.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read image: {meta.path}")

        image = self.transform(image=image)["image"]

        if image.ndim == 2:
            image = image[..., None]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        segment_target, segment_mask = padded_target(
            meta.segment_type,
            meta.label,
        )

        return {
            "image": image,
            "segment_target": torch.tensor(
                segment_target,
                dtype=torch.float32,
            ),
            "segment_mask": torch.tensor(
                segment_mask,
                dtype=torch.float32,
            ),
            "type_target": torch.tensor(
                TYPE_TO_INDEX[meta.segment_type],
                dtype=torch.long,
            ),
            "segment_type": meta.segment_type,
            "has_segment_target": bool(sum(segment_mask)),
            "label": meta.label,
            "display_id": meta.display_id,
            "source_id": meta.source_id,
            "path": str(meta.path),
        }
