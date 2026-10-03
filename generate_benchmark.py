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


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown benchmark report for one model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID, e.g. segment7-2026-10-03-r01",
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
    soft_correct = sum(int(row["correct"]) for row in rows)
    hard_correct = sum(
        int(row.get("hard_prediction") == row["ground_truth"])
        for row in rows
    )

    confidences = [
        value for row in rows
        if (value := safe_float(row.get("confidence"))) is not None
    ]
    margins = [
        value for row in rows
        if (value := safe_float(row.get("margin"))) is not None
    ]

    per_display = defaultdict(lambda: [0, 0])
    for row in rows:
        display_id = row["display_id"]
        per_display[display_id][0] += 1
        per_display[display_id][1] += int(row["correct"])

    lines = [
        f"# Benchmark: {model_id}",
        "",
        "## Overall",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Probabilistic character accuracy | {percent(soft_correct / total)} ({soft_correct}/{total}) |",
        f"| Hard-threshold character accuracy | {percent(hard_correct / total)} ({hard_correct}/{total}) |",
    ]

    if confidences:
        lines.append(
            f"| Mean decoder confidence | {percent(sum(confidences)/len(confidences))} |"
        )
    if margins:
        lines.append(
            f"| Mean confidence margin | {percent(sum(margins)/len(margins))} |"
        )

    lines += [
        "",
        "## Accuracy by display",
        "",
        "| Display | Correct | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ]

    for display_id in sorted(per_display):
        n, c = per_display[display_id]
        lines.append(f"| `{display_id}` | {c} | {n} | {percent(c/n)} |")

    errors = [row for row in rows if int(row["correct"]) == 0]

    lines += ["", "## Misclassifications", ""]
    if not errors:
        lines.append("No misclassified samples.")
    else:
        lines += [
            "| File | Display | Expected | Predicted | Confidence | Margin |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in errors:
            confidence = safe_float(row.get("confidence"))
            margin = safe_float(row.get("margin"))
            lines.append(
                f"| `{row.get('filename','')}` | "
                f"`{row['display_id']}` | "
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
