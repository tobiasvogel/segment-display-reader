import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

from src.config import RAW_DIR, AUGMENTED_DIR, SUPPORTED_EXTENSIONS
from src.dataset import parse_filename
from src.transforms import offline_augmentation_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--copies",
        type=int,
        default=20,
        help="Anzahl Augmentierungen pro Originalbild",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Vorher bestehende augmentierte Bilder löschen",
    )
    args = parser.parse_args()

    src_dir = Path(RAW_DIR)
    dst_dir = Path(AUGMENTED_DIR)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if args.clear:
        for p in dst_dir.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                p.unlink()

    transform = offline_augmentation_transform()

    originals = []
    for path in sorted(src_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if parse_filename(path) is not None:
            originals.append(path)

    if not originals:
        print(f"Keine gültigen Bilder in {src_dir}.")
        return

    total = len(originals) * args.copies
    print(
        f"{len(originals)} Originalbilder × {args.copies} "
        f"= {total} Augmentierungen"
    )

    for path in tqdm(originals):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        for i in range(args.copies):
            aug = transform(image=image)["image"]
            out_name = f"{path.stem}_aug{i:03d}.png"
            cv2.imwrite(str(dst_dir / out_name), aug)

    print(f"Fertig: {dst_dir}")


if __name__ == "__main__":
    main()
