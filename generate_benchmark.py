from collections import defaultdict
from pathlib import Path
import argparse
import csv

from src.config import BENCHMARK_RESULTS_DIR, BENCHMARK_DIR


def percent(value):
    return f"{100.0 * value:.2f}%"


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown benchmark report for one model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID, e.g. segment-2026-10-03-r01",
    )
    args = parser.parse_args()

    model_id = args.model
    csv_path = Path(BENCHMARK_RESULTS_DIR) / f"{model_id}.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"Benchmark CSV not found: {csv_path}\n"
            f"Run: python evaluate.py --model {model_id}"
        )

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("Benchmark CSV contains no rows.")

    total = len(rows)
    type_correct = sum(
        safe_int(row.get("type_correct")) or 0
        for row in rows
    )

    char_rows = [
        row for row in rows
        if safe_int(row.get("correct")) is not None
    ]
    soft_correct = sum(int(row["correct"]) for row in char_rows)
    hard_correct = sum(
        int(row.get("hard_prediction") == row["ground_truth"])
        for row in char_rows
    )
    oracle_correct = sum(
        int(row.get("oracle_type_prediction") == row["ground_truth"])
        for row in char_rows
    )

    confidences = [
        value
        for row in char_rows
        if (value := safe_float(row.get("confidence"))) is not None
    ]
    margins = [
        value
        for row in char_rows
        if (value := safe_float(row.get("margin"))) is not None
    ]

    per_display = defaultdict(
        lambda: {
            "total": 0,
            "type_correct": 0,
            "char_total": 0,
            "char_correct": 0,
        }
    )

    for row in rows:
        stats = per_display[row["display_id"]]
        stats["total"] += 1
        stats["type_correct"] += safe_int(row.get("type_correct")) or 0

        correct = safe_int(row.get("correct"))
        if correct is not None:
            stats["char_total"] += 1
            stats["char_correct"] += correct

    lines = [
        f"# Benchmark: {model_id}",
        "",
        "## Overall",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Segment-type accuracy | {percent(type_correct / total)} ({type_correct}/{total}) |",
    ]

    if char_rows:
        lines += [
            f"| End-to-end probabilistic character accuracy | {percent(soft_correct / len(char_rows))} ({soft_correct}/{len(char_rows)}) |",
            f"| Oracle-type probabilistic character accuracy | {percent(oracle_correct / len(char_rows))} ({oracle_correct}/{len(char_rows)}) |",
            f"| Hard-threshold character accuracy | {percent(hard_correct / len(char_rows))} ({hard_correct}/{len(char_rows)}) |",
        ]

    if confidences:
        lines.append(
            f"| Mean decoder confidence | "
            f"{percent(sum(confidences) / len(confidences))} |"
        )
    if margins:
        lines.append(
            f"| Mean confidence margin | "
            f"{percent(sum(margins) / len(margins))} |"
        )

    lines += [
        "",
        "## Accuracy by display",
        "",
        "| Display | Type | Character | Samples |",
        "|---|---:|---:|---:|",
    ]

    for display_id in sorted(per_display):
        stats = per_display[display_id]
        type_result = percent(
            stats["type_correct"] / stats["total"]
        )

        if stats["char_total"]:
            char_result = (
                f"{percent(stats['char_correct'] / stats['char_total'])} "
                f"({stats['char_correct']}/{stats['char_total']})"
            )
        else:
            char_result = "n/a"

        lines.append(
            f"| `{display_id}` | {type_result} | "
            f"{char_result} | {stats['total']} |"
        )

    errors = [
        row for row in char_rows
        if int(row["correct"]) == 0
    ]

    lines += ["", "## Misclassifications", ""]
    if not errors:
        lines.append("No decodable samples were misclassified.")
    else:
        lines += [
            "| File | Type | Expected | Predicted | Confidence | Margin |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in errors:
            confidence = safe_float(row.get("confidence"))
            margin = safe_float(row.get("margin"))
            type_text = (
                f"{row.get('true_segment_type', '?')}→"
                f"{row.get('predicted_segment_type', '?')}"
            )
            lines.append(
                f"| `{row.get('filename','')}` | "
                f"`{type_text}` | "
                f"`{row['ground_truth']}` | "
                f"`{row['prediction']}` | "
                f"{percent(confidence) if confidence is not None else ''} | "
                f"{percent(margin) if margin is not None else ''} |"
            )

    lines += [
        "",
        f"_Generated from `{csv_path.as_posix()}`._",
        "",
    ]

    out_dir = Path(BENCHMARK_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{model_id}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Benchmark written to: {out_path}")


if __name__ == "__main__":
    main()
