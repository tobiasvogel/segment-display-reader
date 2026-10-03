from collections import defaultdict
from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader

from src.config import (
    RAW_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SEGMENT_THRESHOLD,
)
from src.dataset import collect_samples, SegmentDataset
from src.transforms import eval_transform
from src.model import SegmentCNN
from src.labels import SEGMENT_ORDER, decode_segments
from src.decoder import decode_probabilities


def main():
    samples = collect_samples(RAW_DIR)
    if not samples:
        raise RuntimeError("No raw images found.")

    ds = SegmentDataset(samples, eval_transform())
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path(MODEL_DIR) / "best_model.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(
            "No model found. Run python train.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SegmentCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    total = 0
    soft_correct = 0
    hard_correct = 0
    seg_correct = 0
    seg_total = 0
    exact_bits_correct = 0

    per_display = defaultdict(lambda: [0, 0])
    rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            probs = torch.sigmoid(model(images))
            bits = (probs >= SEGMENT_THRESHOLD).int()
            target_bits = targets.int()

            seg_correct += int((bits == target_bits).sum().item())
            seg_total += int(target_bits.numel())

            exact_bits_correct += int(
                torch.all(bits == target_bits, dim=1).sum().item()
            )

            for i in range(images.size(0)):
                pred_probs = probs[i].cpu().tolist()
                pred_bits = bits[i].cpu().tolist()

                decoded = decode_probabilities(pred_probs)
                soft_pred = decoded["char"]
                hard_pred = decode_segments(pred_bits)

                true = batch["label"][i]
                display_id = batch["display_id"][i]
                path = batch["path"][i]

                correct = soft_pred == true
                total += 1
                soft_correct += int(correct)
                hard_correct += int(hard_pred == true)

                per_display[display_id][0] += 1
                per_display[display_id][1] += int(correct)

                row = {
                    "filename": Path(path).name,
                    "display_id": display_id,
                    "ground_truth": true,
                    "prediction": soft_pred,
                    "correct": int(correct),
                    "confidence": f"{decoded['confidence']:.6f}",
                    "second_prediction": decoded["second_char"],
                    "second_confidence": (
                        f"{decoded['second_confidence']:.6f}"
                    ),
                    "margin": f"{decoded['margin']:.6f}",
                    "hard_prediction": hard_pred,
                    "hard_pattern": "".join(map(str, pred_bits)),
                }

                for j, seg in enumerate(SEGMENT_ORDER):
                    row[f"{seg}_prob"] = (
                        f"{float(probs[i, j].cpu()):.6f}"
                    )

                rows.append(row)

    print(
        f"Probabilistic character accuracy: "
        f"{100*soft_correct/max(1, total):.2f}% "
        f"({soft_correct}/{total})"
    )
    print(
        f"Hard-threshold character accuracy: "
        f"{100*hard_correct/max(1, total):.2f}% "
        f"({hard_correct}/{total})"
    )
    print(
        f"Exact 7-bit accuracy: "
        f"{100*exact_bits_correct/max(1, total):.2f}%"
    )
    print(
        f"Segment accuracy: "
        f"{100*seg_correct/max(1, seg_total):.2f}%"
    )

    print("\nProbabilistic accuracy per display:")
    for display_id in sorted(per_display):
        display_total, display_correct = per_display[display_id]
        print(
            f"{display_id}: "
            f"{100*display_correct/display_total:.2f}% "
            f"({display_correct}/{display_total})"
        )

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "cnn_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
