from pathlib import Path
from collections import defaultdict
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


def main():
    samples = collect_samples(RAW_DIR)
    if not samples:
        raise RuntimeError("Keine Rohbilder gefunden.")

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
            "Kein Modell gefunden. Zuerst python train.py ausführen."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SegmentCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    overall_total = 0
    overall_correct = 0
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
                pred_bits = bits[i].cpu().tolist()
                pred = decode_segments(pred_bits)
                true = batch["label"][i]
                display_id = batch["display_id"][i]
                path = batch["path"][i]

                correct = pred == true
                overall_total += 1
                overall_correct += int(correct)

                per_display[display_id][0] += 1
                per_display[display_id][1] += int(correct)

                row = {
                    "filename": Path(path).name,
                    "display_id": display_id,
                    "ground_truth": true,
                    "prediction": pred,
                    "correct": int(correct),
                    "pattern": "".join(map(str, pred_bits)),
                }

                for j, seg in enumerate(SEGMENT_ORDER):
                    row[f"{seg}_prob"] = (
                        f"{float(probs[i, j].cpu()):.6f}"
                    )

                rows.append(row)

    print(
        f"Character accuracy: "
        f"{100*overall_correct/max(1, overall_total):.2f}% "
        f"({overall_correct}/{overall_total})"
    )

    print(
        f"Exact 7-bit accuracy: "
        f"{100*exact_bits_correct/max(1, overall_total):.2f}%"
    )

    print(
        f"Segment accuracy: "
        f"{100*seg_correct/max(1, seg_total):.2f}%"
    )

    print("\nPer display:")
    for display_id in sorted(per_display):
        total, correct = per_display[display_id]
        print(
            f"{display_id}: "
            f"{100*correct/total:.2f}% ({correct}/{total})"
        )

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "cnn_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
