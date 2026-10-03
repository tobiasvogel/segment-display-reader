from datetime import datetime
import json
from pathlib import Path

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


def next_model_id():
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    date = datetime.now().astimezone().strftime("%Y-%m-%d")
    prefix = f"{MODEL_PREFIX}-{date}-r"

    runs = []
    for path in model_dir.glob(f"{prefix}*.pt"):
        suffix = path.stem.removeprefix(prefix)
        if suffix.isdigit():
            runs.append(int(suffix))

    return f"{prefix}{max(runs, default=0) + 1:02d}"


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
