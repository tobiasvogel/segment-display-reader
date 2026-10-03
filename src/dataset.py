from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import CHAR_TO_SEGMENTS
from .config import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class SampleMeta:
    path: Path
    display_id: str
    label: str
    index: int

    @property
    def source_id(self):
        """Stable ID shared by an original image and all of its augmentations."""
        return f"{self.display_id}:{self.label}:{self.index}"


# Accepts original files:
# d0_4_001.png
#
# and offline augmentations:
# d0_4_001_aug012.png
_FILENAME_RE = re.compile(
    r"^d(?P<display>[A-Za-z0-9-]+)_"
    r"(?P<label>[0-9-])_"
    r"(?P<index>\d+)"
    r"(?:_aug\d+)?"
    r"\.[^.]+$"
)


def parse_filename(path):
    path = Path(path)
    match = _FILENAME_RE.match(path.name)
    if not match:
        return None

    label = match.group("label")
    if label not in CHAR_TO_SEGMENTS:
        return None

    return SampleMeta(
        path=path,
        display_id=f"d{match.group('display')}",
        label=label,
        index=int(match.group("index")),
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
            if meta is not None:
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

        # HWC -> CHW; grayscale has one channel.
        if image.ndim == 2:
            image = image[..., None]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        target = torch.tensor(
            CHAR_TO_SEGMENTS[meta.label],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "target": target,
            "label": meta.label,
            "display_id": meta.display_id,
            "source_id": meta.source_id,
            "path": str(meta.path),
        }
