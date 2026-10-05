#!/usr/bin/env bash
set -euo pipefail

TRAINING_REPO="tobiasvogel/segment-display-trainingdata"
COPIES=20
PYTHON_OVERRIDE=""

usage() {
    cat <<'EOF'
Usage:
  ./prepare_training_data.sh [--copies N] [--python PATH]

Downloads the latest labeled training data from the private training-data
repository into data/raw/, clears/rebuilds data/augmented/, and leaves the
project ready for train.py.

Options:
  --copies N          Number of offline augmentations per original image (default: 20)
  --python PATH       Python 3 executable to use, e.g. /opt/homebrew/bin/python3
  --python-executable PATH
                      Alias for --python
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --copies)
            if [[ $# -lt 2 ]]; then
                echo "Error: --copies requires a value." >&2
                exit 2
            fi
            COPIES="$2"
            shift 2
            ;;
        --python|--python-executable)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a path." >&2
                exit 2
            fi
            PYTHON_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "$COPIES" =~ ^[0-9]+$ ]] || [[ "$COPIES" -lt 1 ]]; then
    echo "Error: --copies must be an integer >= 1." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "augment_dataset.py" ]] || [[ ! -f "train.py" ]]; then
    echo "Error: run this script from the segment-display-reader repository." >&2
    exit 1
fi

if [[ -n "$PYTHON_OVERRIDE" ]]; then
    if [[ ! -x "$PYTHON_OVERRIDE" ]]; then
        echo "Error: Python executable is not executable or does not exist:" >&2
        echo "  $PYTHON_OVERRIDE" >&2
        exit 1
    fi
    PYTHON="$PYTHON_OVERRIDE"
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "Error: Python was not found." >&2
    exit 1
fi

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info.major == 3 else 1)'; then
    echo "Error: selected Python executable is not Python 3:" >&2
    echo "  $PYTHON" >&2
    exit 1
fi

echo "Using Python: $PYTHON"

TMP_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

TRAINING_DIR="$TMP_DIR/trainingdata"

echo "Fetching latest training data from $TRAINING_REPO ..."

if command -v gh >/dev/null 2>&1; then
    if ! gh auth status >/dev/null 2>&1; then
        echo "Error: GitHub CLI is installed but not authenticated." >&2
        echo "Run: gh auth login" >&2
        exit 1
    fi

    gh repo clone "$TRAINING_REPO" "$TRAINING_DIR" -- --depth 1
else
    echo "GitHub CLI not found; falling back to git clone."
    echo "Your Git credentials/SSH key must have access to the private repository."

    git clone --depth 1         "git@github.com:${TRAINING_REPO}.git"         "$TRAINING_DIR"
fi

SOURCE_DIR="$TRAINING_DIR/images_labeled"
RAW_DIR="$SCRIPT_DIR/data/raw"

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: images_labeled/ was not found in the training-data repository." >&2
    exit 1
fi

mkdir -p "$RAW_DIR"

echo "Syncing labeled images into data/raw/ ..."

# The private repository is the source of truth for data/raw.
# Remove old image files before copying so deleted/renamed training samples
# do not accidentally remain in the local training set.
find "$RAW_DIR" -maxdepth 1 -type f \(     -iname '*.png' -o     -iname '*.jpg' -o     -iname '*.jpeg' -o     -iname '*.bmp' -o     -iname '*.webp' \) -delete

find "$SOURCE_DIR" -maxdepth 1 -type f \(     -iname '*.png' -o     -iname '*.jpg' -o     -iname '*.jpeg' -o     -iname '*.bmp' -o     -iname '*.webp' \) -exec cp {} "$RAW_DIR/" \;

RAW_COUNT="$(find "$RAW_DIR" -maxdepth 1 -type f \(     -iname '*.png' -o     -iname '*.jpg' -o     -iname '*.jpeg' -o     -iname '*.bmp' -o     -iname '*.webp' \) | wc -l | tr -d ' ')"

echo "Raw training images: $RAW_COUNT"

if [[ "$RAW_COUNT" -eq 0 ]]; then
    echo "Error: no labeled images were copied." >&2
    exit 1
fi

echo "Generating augmented dataset ($COPIES copies per source image) ..."
"$PYTHON" augment_dataset.py --clear --copies "$COPIES"

echo
echo "Training data is ready."
echo "Next step:"
echo "  $PYTHON train.py"
