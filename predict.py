from pathlib import Path
import argparse

import cv2
import numpy as np
import torch

from src.config import SEGMENT_TYPES, SEGMENT_THRESHOLD
from src.decoder import decode_probabilities
from src.labels import INDEX_TO_TYPE, SEGMENT_NAMES, hard_decode
from src.model import SegmentCNN
from src.model_registry import resolve_model_path
from src.transforms import eval_transform


def parse_crop(value):
    try:
        parts = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Crop must be x1,y1,x2,y2 using integer pixel coordinates."
        ) from exc

    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Crop must contain exactly four values: x1,y1,x2,y2."
        )

    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError(
            "Crop requires x2 > x1 and y2 > y1."
        )

    return tuple(parts)


def prepare_image(image):
    transformed = eval_transform()(image=image)["image"]

    if transformed.ndim == 2:
        transformed = transformed[..., None]

    transformed = (
        np.transpose(transformed, (2, 0, 1))
        .astype(np.float32)
        / 255.0
    )

    return torch.from_numpy(transformed)


def split_positions_equal(image, digits):
    if digits < 1:
        raise ValueError("--digits must be >= 1.")

    height, width = image.shape[:2]
    boundaries = np.linspace(0, width, digits + 1).round().astype(int)
    return _crops_from_boundaries(image, boundaries)


def _active_x_runs(image):
    # Otsu thresholding works for bright-on-dark and dark-on-bright displays.
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Treat the minority class as foreground so polarity is handled
    # automatically for typical display crops.
    if np.mean(binary > 0) > 0.5:
        binary = 255 - binary

    height = image.shape[0]
    min_foreground_pixels = max(1, int(round(height * 0.01)))
    active = (
        (binary > 0).sum(axis=0) >= min_foreground_pixels
    )

    runs = []
    start = None

    for x, is_active in enumerate(active):
        if is_active and start is None:
            start = x

        at_end = x == len(active) - 1
        if start is not None and (not is_active or at_end):
            end = x if not is_active else x + 1
            runs.append((start, end))
            start = None

    return runs


def _merge_closest_runs(runs, target_count):
    runs = list(runs)

    while len(runs) > target_count:
        gaps = [
            runs[i + 1][0] - runs[i][1]
            for i in range(len(runs) - 1)
        ]
        merge_at = int(np.argmin(gaps))

        merged = (
            runs[merge_at][0],
            runs[merge_at + 1][1],
        )
        runs[merge_at:merge_at + 2] = [merged]

    return runs


def _crops_from_boundaries(image, boundaries):
    height = image.shape[0]
    crops = []

    for i in range(len(boundaries) - 1):
        x1 = int(boundaries[i])
        x2 = int(boundaries[i + 1])
        crop = image[0:height, x1:x2]

        if crop.size == 0:
            raise RuntimeError(
                f"Position {i + 1} produced an empty crop."
            )

        crops.append((crop, (x1, 0, x2, height)))

    return crops


def split_positions_auto(image, digits):
    """
    Split at whitespace between detected character extents.

    This preserves each character's position inside its display cell. That is
    especially important for narrow characters such as "1", whose lit segments
    sit on the right side of a 7-segment cell.
    """
    if digits < 1:
        raise ValueError("--digits must be >= 1.")

    if digits == 1:
        return [(image, (0, 0, image.shape[1], image.shape[0]))]

    runs = _active_x_runs(image)

    if len(runs) > digits:
        runs = _merge_closest_runs(runs, digits)

    if len(runs) != digits:
        print(
            "Warning: automatic split found "
            f"{len(runs)} character region(s), expected {digits}; "
            "falling back to equal-width splitting."
        )
        return split_positions_equal(image, digits)

    boundaries = [0]
    for left, right in zip(runs[:-1], runs[1:]):
        gap_midpoint = int(round((left[1] + right[0]) / 2.0))
        boundaries.append(gap_midpoint)
    boundaries.append(image.shape[1])

    return _crops_from_boundaries(image, boundaries)


def resolve_common_type(type_logits, forced_type=None):
    if forced_type is not None:
        return forced_type, 1.0

    # A physical multi-position display normally has one segment family.
    # Sum log probabilities across positions to obtain one display-level type.
    type_log_probs = torch.log_softmax(type_logits, dim=1)
    combined = type_log_probs.sum(dim=0)
    index = int(combined.argmax().item())
    segment_type = INDEX_TO_TYPE[index]

    combined_probs = torch.softmax(combined, dim=0)
    confidence = float(combined_probs[index].cpu())

    return segment_type, confidence


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run one image through a trained segment-display model. "
            "Use --digits for multi-position displays."
        )
    )
    parser.add_argument(
        "image",
        help="Path to a single-character or complete display image.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID, filename, or path. Defaults to the active model.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=1,
        help="Number of display positions in the image. Default: 1.",
    )
    parser.add_argument(
        "--split",
        choices=("auto", "equal"),
        default="auto",
        help=(
            "How to split multi-position displays. 'auto' uses whitespace "
            "between detected character regions and preserves character "
            "geometry; 'equal' uses fixed-width cells. Default: auto."
        ),
    )
    parser.add_argument(
        "--type",
        dest="forced_type",
        type=int,
        choices=SEGMENT_TYPES,
        default=None,
        help=(
            "Override automatic segment-type detection with 7, 13, 14, or 16."
        ),
    )
    parser.add_argument(
        "--crop",
        type=parse_crop,
        default=None,
        metavar="X1,Y1,X2,Y2",
        help=(
            "Optional pixel crop applied before splitting into positions."
        ),
    )
    parser.add_argument(
        "--save-crops",
        default=None,
        metavar="DIR",
        help="Optionally save the generated per-position crops for inspection.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    if args.crop is not None:
        x1, y1, x2, y2 = args.crop
        height, width = image.shape[:2]

        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError(
                f"--crop {args.crop} exceeds image bounds "
                f"{width}x{height}."
            )

        image = image[y1:y2, x1:x2]

    if args.split == "auto":
        positions = split_positions_auto(image, args.digits)
    else:
        positions = split_positions_equal(image, args.digits)

    if args.save_crops:
        crop_dir = Path(args.save_crops)
        crop_dir.mkdir(parents=True, exist_ok=True)

        for i, (crop, _) in enumerate(positions, start=1):
            cv2.imwrite(
                str(crop_dir / f"{image_path.stem}_pos{i:02d}.png"),
                crop,
            )

    tensors = torch.stack(
        [prepare_image(crop) for crop, _ in positions],
        dim=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = resolve_model_path(args.model)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    architecture = checkpoint.get("architecture")
    if architecture != "segment-multitask-v2":
        raise RuntimeError(
            "This prediction script requires a segment-multitask-v2 "
            "checkpoint. Retrain with the current train.py."
        )

    model_id = checkpoint.get("model_id", checkpoint_path.stem)

    model = SegmentCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        outputs = model(tensors.to(device))
        type_logits = outputs["type_logits"].cpu()
        segment_probs = torch.sigmoid(
            outputs["segment_logits"]
        ).cpu()

    segment_type, type_confidence = resolve_common_type(
        type_logits,
        forced_type=args.forced_type,
    )

    print(f"Model: {model_id}")
    if args.forced_type is None:
        print(
            f"Segment type: {segment_type} "
            f"({100 * type_confidence:.2f}% display-level confidence)"
        )
    else:
        print(f"Segment type: {segment_type} (forced)")

    print(f"Positions: {len(positions)}")
    print()

    decoded_chars = []

    for i, ((_, bounds), probs_tensor) in enumerate(
        zip(positions, segment_probs),
        start=1,
    ):
        probs = probs_tensor.tolist()
        decoded = decode_probabilities(segment_type, probs)
        bits = (
            probs_tensor >= SEGMENT_THRESHOLD
        ).int().tolist()
        hard_pred = hard_decode(segment_type, bits)

        decoded_chars.append(decoded["char"])

        x1, y1, x2, y2 = bounds
        print(
            f"[{i:02d}] x={x1}:{x2}  "
            f"prediction={decoded['char']}  "
            f"confidence={100*decoded['confidence']:.2f}%  "
            f"second={decoded['second_char'] or '-'}  "
            f"margin={100*decoded['margin']:.2f}%  "
            f"hard={hard_pred}"
        )

        active_count = min(segment_type, len(SEGMENT_NAMES))
        segment_text = " ".join(
            f"{SEGMENT_NAMES[j]}={probs[j]:.3f}"
            for j in range(active_count)
        )
        print(f"     {segment_text}")

    print()
    print(f"Result: {''.join(decoded_chars)}")


if __name__ == "__main__":
    main()
