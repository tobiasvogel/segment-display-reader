from datetime import datetime
import json
from pathlib import Path
import re

from .config import MODEL_DIR, MODEL_PREFIX


REGISTRY_FILENAME = "model_registry.json"


def registry_path():
    return Path(MODEL_DIR) / REGISTRY_FILENAME


def load_registry():
    path = registry_path()
    if not path.exists():
        return {"active": None, "models": {}}

    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("active", None)
    data.setdefault("models", {})
    return data


def save_registry(registry):
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _model_id(date, run):
    return f"{MODEL_PREFIX}-{date}-r{run:02d}"


def next_model_id(run=None):
    """
    Return a model ID for today's date.

    Without an explicit run number, determine the next run from both:
      - versioned .pt files in models/
      - model IDs already recorded in model_registry.json

    Passing run=N overrides automatic allocation, e.g. --run 4 -> r04.
    Existing checkpoint files are never overwritten accidentally.
    """
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    date = datetime.now().astimezone().strftime("%Y-%m-%d")
    prefix = f"{MODEL_PREFIX}-{date}-r"

    if run is not None:
        run = int(run)
        if run < 1:
            raise ValueError("Run number must be >= 1.")

        model_id = _model_id(date, run)
        checkpoint_path = model_dir / f"{model_id}.pt"

        if checkpoint_path.exists():
            raise FileExistsError(
                f"Checkpoint already exists: {checkpoint_path}. "
                "Choose another --run value."
            )

        return model_id

    runs = set()

    # Check files that are present locally.
    pattern = re.compile(
        rf"^{re.escape(prefix)}(?P<run>\d+)\.pt$"
    )
    for path in model_dir.glob(f"{prefix}*.pt"):
        match = pattern.match(path.name)
        if match:
            runs.add(int(match.group("run")))

    # Also check registry history, because older checkpoints may no longer be
    # present on the current machine.
    registry = load_registry()
    registry_pattern = re.compile(
        rf"^{re.escape(prefix)}(?P<run>\d+)$"
    )
    for model_id in registry.get("models", {}):
        match = registry_pattern.match(model_id)
        if match:
            runs.add(int(match.group("run")))

    return _model_id(date, max(runs, default=0) + 1)


def register_model(model_id, checkpoint_path, metadata):
    registry = load_registry()
    registry["active"] = checkpoint_path.name
    registry["models"][model_id] = {
        "file": checkpoint_path.name,
        **metadata,
    }
    save_registry(registry)


def resolve_model_path(model=None):
    model_dir = Path(MODEL_DIR)

    if model:
        candidate = Path(model)
        if candidate.exists():
            return candidate

        candidate = model_dir / model
        if candidate.exists():
            return candidate

        if not str(model).endswith(".pt"):
            candidate = model_dir / f"{model}.pt"
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Model not found: {model}")

    registry = load_registry()
    active = registry.get("active")
    if active:
        candidate = model_dir / active
        if candidate.exists():
            return candidate

    legacy = model_dir / "best_model.pt"
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        "No active model found. Run python train.py first or pass --model."
    )
