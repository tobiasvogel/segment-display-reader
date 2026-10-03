from collections import defaultdict
from pathlib import Path
import argparse
import csv

import torch
from torch.utils.data import DataLoader

from src.config import (
    RAW_DIR,
    BENCHMARK_RESULTS_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SEGMENT_THRESHOLD,
)
from src.dataset import collect_samples, SegmentDataset
from src.transforms import eval_transform
from src.model import SegmentCNN
from src.labels import (
    INDEX_TO_TYPE,
    SEGMENT_NAMES,
    get_charset,
    hard_decode,
)
from src.decoder import decode_probabilities
from src.model_registry import resolve_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID, filename, or path. Defaults to the active model.",
    )
    args = parser.parse_args()

    samples = collect_samples(RAW_DIR)
    if not samples:
        raise RuntimeError("No raw images found.")

    loader = DataLoader(
        SegmentDataset(samples, eval_transform()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = resolve_model_path(args.model)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_id = checkpoint.get("model_id", checkpoint_path.stem)
    print(f"Evaluating model: {model_id}")

    architecture = checkpoint.get("architecture")
    if architecture != "segment-multitask-v1":
        raise RuntimeError(
            "This checkpoint predates the multi-task architecture or uses an "
            f"unsupported architecture ({architecture!r}). Retrain with the "
            "current train.py before evaluating it."
        )

    model = SegmentCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    total = 0
    type_correct = 0
    char_total = 0
    soft_correct = 0
    oracle_soft_correct = 0
    hard_correct = 0
    exact_total = 0
    exact_correct = 0
    seg_correct = 0
    seg_total = 0

    per_display = defaultdict(
        lambda: {
            "total": 0,
            "type_correct": 0,
            "char_total": 0,
            "char_correct": 0,
        }
    )
    rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)

            type_probs = torch.softmax(outputs["type_logits"], dim=1)
            predicted_type_indices = type_probs.argmax(dim=1)

            segment_probs = torch.sigmoid(outputs["segment_logits"])
            bits = (segment_probs >= SEGMENT_THRESHOLD).int()

            targets = batch["segment_target"].to(device).int()
            mask = batch["segment_mask"].to(device).bool()
            type_targets = batch["type_target"].to(device)

            n = images.size(0)
            total += n
            type_correct += int(
                (predicted_type_indices == type_targets).sum().item()
            )

            seg_correct += int(((bits == targets) & mask).sum().item())
            seg_total += int(mask.sum().item())

            for i in range(n):
                true_type = int(batch["segment_type"][i])
                pred_type_idx = int(predicted_type_indices[i].item())
                pred_type = INDEX_TO_TYPE[pred_type_idx]
                pred_type_confidence = float(
                    type_probs[i, pred_type_idx].cpu()
                )

                pred_probs = segment_probs[i].cpu().tolist()
                pred_bits = bits[i].cpu().tolist()

                true_label = batch["label"][i]
                display_id = batch["display_id"][i]
                path = batch["path"][i]

                decoded = decode_probabilities(pred_type, pred_probs)
                oracle_decoded = decode_probabilities(true_type, pred_probs)
                hard_pred = hard_decode(pred_type, pred_bits)

                type_ok = pred_type == true_type

                display_stats = per_display[display_id]
                display_stats["total"] += 1
                display_stats["type_correct"] += int(type_ok)

                has_charset = bool(get_charset(true_type))
                if has_charset:
                    char_total += 1
                    display_stats["char_total"] += 1

                    soft_ok = decoded["char"] == true_label
                    oracle_ok = oracle_decoded["char"] == true_label
                    hard_ok = hard_pred == true_label

                    soft_correct += int(soft_ok)
                    oracle_soft_correct += int(oracle_ok)
                    hard_correct += int(hard_ok)
                    display_stats["char_correct"] += int(soft_ok)
                else:
                    soft_ok = False

                if mask[i].any():
                    exact_total += 1
                    exact_correct += int(
                        torch.all(
                            bits[i][mask[i]] == targets[i][mask[i]]
                        ).item()
                    )

                row = {
                    "model_id": model_id,
                    "filename": Path(path).name,
                    "display_id": display_id,
                    "ground_truth": true_label,
                    "true_segment_type": true_type,
                    "predicted_segment_type": pred_type,
                    "type_correct": int(type_ok),
                    "type_confidence": f"{pred_type_confidence:.6f}",
                    "prediction": decoded["char"],
                    "correct": int(soft_ok) if has_charset else "",
                    "confidence": f"{decoded['confidence']:.6f}",
                    "second_prediction": decoded["second_char"],
                    "second_confidence": (
                        f"{decoded['second_confidence']:.6f}"
                    ),
                    "margin": f"{decoded['margin']:.6f}",
                    "oracle_type_prediction": oracle_decoded["char"],
                    "hard_prediction": hard_pred,
                    "hard_pattern": "".join(
                        map(str, pred_bits[:pred_type])
                    ),
                }

                for j, name in enumerate(SEGMENT_NAMES):
                    row[f"{name}_prob"] = (
                        f"{float(segment_probs[i, j].cpu()):.6f}"
                    )

                rows.append(row)

    print(
        f"Segment-type accuracy: "
        f"{100*type_correct/max(1,total):.2f}% "
        f"({type_correct}/{total})"
    )

    if char_total:
        print(
            f"End-to-end probabilistic character accuracy: "
            f"{100*soft_correct/char_total:.2f}% "
            f"({soft_correct}/{char_total})"
        )
        print(
            f"Oracle-type probabilistic character accuracy: "
            f"{100*oracle_soft_correct/char_total:.2f}% "
            f"({oracle_soft_correct}/{char_total})"
        )
        print(
            f"Hard-threshold character accuracy: "
            f"{100*hard_correct/char_total:.2f}% "
            f"({hard_correct}/{char_total})"
        )

    if exact_total:
        print(
            f"Exact segment-pattern accuracy: "
            f"{100*exact_correct/exact_total:.2f}% "
            f"({exact_correct}/{exact_total})"
        )

    if seg_total:
        print(
            f"Masked segment accuracy: "
            f"{100*seg_correct/seg_total:.2f}%"
        )

    print("\nPer display:")
    for display_id in sorted(per_display):
        stats = per_display[display_id]
        type_acc = stats["type_correct"] / stats["total"]

        if stats["char_total"]:
            char_result = (
                f", char={100*stats['char_correct']/stats['char_total']:.2f}% "
                f"({stats['char_correct']}/{stats['char_total']})"
            )
        else:
            char_result = ", char=n/a"

        print(
            f"{display_id}: type={100*type_acc:.2f}% "
            f"({stats['type_correct']}/{stats['total']})"
            f"{char_result}"
        )

    out_dir = Path(BENCHMARK_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{model_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
