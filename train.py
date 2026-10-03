from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    RAW_DIR, AUGMENTED_DIR, MODEL_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, NUM_WORKERS, RANDOM_SEED, SEGMENT_THRESHOLD,
    EARLY_STOPPING_PATIENCE, IMAGE_WIDTH, IMAGE_HEIGHT,
)
from src.dataset import collect_samples, SegmentDataset
from src.transforms import train_transform, eval_transform
from src.split import split_samples
from src.model import SegmentCNN
from src.decoder import decode_probabilities
from src.model_registry import next_model_id, register_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total = char_correct = exact_correct = seg_correct = seg_total = 0
    loss_sum = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.sigmoid(logits)
        bits = (probs >= SEGMENT_THRESHOLD).int()
        target_bits = targets.int()

        n = images.size(0)
        total += n
        loss_sum += float(loss.item()) * n
        exact_correct += int(torch.all(bits == target_bits, dim=1).sum().item())
        seg_correct += int((bits == target_bits).sum().item())
        seg_total += int(target_bits.numel())

        for pred_probs, true_label in zip(probs.cpu().tolist(), batch["label"]):
            char_correct += int(
                decode_probabilities(pred_probs)["char"] == true_label
            )

    return {
        "loss": loss_sum / max(1, total),
        "char_acc": char_correct / max(1, total),
        "exact_bits_acc": exact_correct / max(1, total),
        "segment_acc": seg_correct / max(1, seg_total),
    }


def main():
    seed_everything(RANDOM_SEED)
    samples = collect_samples(RAW_DIR, AUGMENTED_DIR)
    if len(samples) < 2:
        raise RuntimeError("Too few samples for training.")

    train_samples, val_samples = split_samples(samples)
    train_sources = len({s.source_id for s in train_samples})
    val_sources = len({s.source_id for s in val_samples})

    print(f"Train: {len(train_samples)} files from {train_sources} source images")
    print(f"Validation: {len(val_samples)} files from {val_sources} source images")

    train_loader = DataLoader(
        SegmentDataset(train_samples, train_transform()),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        SegmentDataset(val_samples, eval_transform()),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SegmentCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_id = next_model_id()
    checkpoint_path = model_dir / f"{model_id}.pt"
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"Model ID: {model_id}")

    best_val = -1.0
    best_epoch = None
    best_metrics = None
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = seen = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS}", leave=False)
        for batch in bar:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(1, seen)
        metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"char_acc={metrics['char_acc']*100:6.2f}% "
            f"exact_bits={metrics['exact_bits_acc']*100:6.2f}% "
            f"segment_acc={metrics['segment_acc']*100:6.2f}%"
        )

        if metrics["char_acc"] > best_val:
            best_val = metrics["char_acc"]
            best_epoch = epoch
            best_metrics = metrics
            stale = 0

            checkpoint = {
                "model_state": model.state_dict(),
                "model_id": model_id,
                "created_at": created_at,
                "best_epoch": best_epoch,
                "val_metrics": best_metrics,
                "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
                "segment_threshold": SEGMENT_THRESHOLD,
                "decoder": "probabilistic-v1",
                "train_source_images": train_sources,
                "validation_source_images": val_sources,
            }
            torch.save(checkpoint, checkpoint_path)
            register_model(
                model_id, checkpoint_path,
                {
                    "created_at": created_at,
                    "best_epoch": best_epoch,
                    "val_metrics": best_metrics,
                    "train_source_images": train_sources,
                    "validation_source_images": val_sources,
                    "decoder": "probabilistic-v1",
                },
            )
            print(f"  -> new best checkpoint: {checkpoint_path}")
        else:
            stale += 1

        if stale >= EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping: validation character accuracy did not improve "
                f"for {EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    print(f"\nModel: {checkpoint_path}")
    print(f"Best epoch: {best_epoch}")
    if best_metrics:
        print(f"Best validation character accuracy: {best_metrics['char_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
