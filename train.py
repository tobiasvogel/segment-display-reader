from datetime import datetime
from pathlib import Path
import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    RAW_DIR,
    AUGMENTED_DIR,
    MODEL_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    RANDOM_SEED,
    SEGMENT_THRESHOLD,
    EARLY_STOPPING_PATIENCE,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    TYPE_LOSS_WEIGHT,
    SEGMENT_LOSS_WEIGHT,
    SEGMENT_TYPES,
)
from src.dataset import collect_samples, SegmentDataset
from src.transforms import train_transform, eval_transform
from src.split import split_samples
from src.model import SegmentCNN
from src.decoder import decode_probabilities
from src.labels import INDEX_TO_TYPE, get_charset
from src.model_registry import next_model_id, register_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multitask_loss(outputs, batch, type_criterion, segment_criterion):
    type_targets = batch["type_target"].to(outputs["type_logits"].device)
    segment_targets = batch["segment_target"].to(
        outputs["segment_logits"].device
    )
    segment_mask = batch["segment_mask"].to(outputs["segment_logits"].device)

    type_loss = type_criterion(outputs["type_logits"], type_targets)

    per_segment = segment_criterion(
        outputs["segment_logits"],
        segment_targets,
    )
    masked_sum = (per_segment * segment_mask).sum()
    mask_count = segment_mask.sum()

    if mask_count.item() > 0:
        segment_loss = masked_sum / mask_count
    else:
        segment_loss = outputs["segment_logits"].sum() * 0.0

    total = (
        TYPE_LOSS_WEIGHT * type_loss
        + SEGMENT_LOSS_WEIGHT * segment_loss
    )
    return total, type_loss, segment_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    type_criterion = nn.CrossEntropyLoss()
    segment_criterion = nn.BCEWithLogitsLoss(reduction="none")

    sample_total = 0
    type_correct = 0
    char_total = 0
    char_correct = 0
    oracle_char_correct = 0
    exact_total = 0
    exact_correct = 0
    segment_correct = 0
    segment_total = 0

    loss_sum = 0.0
    type_loss_sum = 0.0
    segment_loss_sum = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        outputs = model(images)

        loss, type_loss, segment_loss = multitask_loss(
            outputs,
            batch,
            type_criterion,
            segment_criterion,
        )

        probs = torch.sigmoid(outputs["segment_logits"])
        bits = (probs >= SEGMENT_THRESHOLD).int()
        predicted_type_indices = outputs["type_logits"].argmax(dim=1)

        targets = batch["segment_target"].to(device).int()
        mask = batch["segment_mask"].to(device).bool()
        type_targets = batch["type_target"].to(device)

        n = images.size(0)
        sample_total += n
        loss_sum += float(loss.item()) * n
        type_loss_sum += float(type_loss.item()) * n
        segment_loss_sum += float(segment_loss.item()) * n

        type_correct += int(
            (predicted_type_indices == type_targets).sum().item()
        )

        segment_correct += int(((bits == targets) & mask).sum().item())
        segment_total += int(mask.sum().item())

        for i in range(n):
            true_type = int(batch["segment_type"][i])
            pred_type = INDEX_TO_TYPE[int(predicted_type_indices[i].item())]
            true_label = batch["label"][i]
            pred_probs = probs[i].cpu().tolist()

            if mask[i].any():
                exact_total += 1
                exact_correct += int(
                    torch.all(bits[i][mask[i]] == targets[i][mask[i]]).item()
                )

            if get_charset(true_type):
                char_total += 1
                pred_char = decode_probabilities(
                    pred_type,
                    pred_probs,
                )["char"]
                oracle_char = decode_probabilities(
                    true_type,
                    pred_probs,
                )["char"]
                char_correct += int(pred_char == true_label)
                oracle_char_correct += int(oracle_char == true_label)

    return {
        "loss": loss_sum / max(1, sample_total),
        "type_loss": type_loss_sum / max(1, sample_total),
        "segment_loss": segment_loss_sum / max(1, sample_total),
        "type_acc": type_correct / max(1, sample_total),
        "char_acc": char_correct / max(1, char_total),
        "oracle_type_char_acc": oracle_char_correct / max(1, char_total),
        "exact_bits_acc": exact_correct / max(1, exact_total),
        "segment_acc": segment_correct / max(1, segment_total),
        "char_samples": char_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help=(
            "Override today's run number, e.g. --run 4 creates r04. "
            "If omitted, the next run is determined automatically."
        ),
    )
    args = parser.parse_args()

    seed_everything(RANDOM_SEED)

    samples = collect_samples(RAW_DIR, AUGMENTED_DIR)
    if len(samples) < 2:
        raise RuntimeError("Too few samples for training.")

    train_samples, val_samples = split_samples(samples)
    train_sources = len({s.source_id for s in train_samples})
    val_sources = len({s.source_id for s in val_samples})

    print(
        f"Train: {len(train_samples)} files "
        f"from {train_sources} source images"
    )
    print(
        f"Validation: {len(val_samples)} files "
        f"from {val_sources} source images"
    )

    train_loader = DataLoader(
        SegmentDataset(train_samples, train_transform()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        SegmentDataset(val_samples, eval_transform()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model = SegmentCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    type_criterion = nn.CrossEntropyLoss()
    segment_criterion = nn.BCEWithLogitsLoss(reduction="none")

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_id = next_model_id(run=args.run)
    checkpoint_path = model_dir / f"{model_id}.pt"
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"Model ID: {model_id}")

    best_val = -1.0
    best_epoch = None
    best_metrics = None
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss = 0.0
        seen = 0

        bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{EPOCHS}",
            leave=False,
        )

        for batch in bar:
            images = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss, type_loss, segment_loss = multitask_loss(
                outputs,
                batch,
                type_criterion,
                segment_criterion,
            )

            loss.backward()
            optimizer.step()

            n = images.size(0)
            train_loss += float(loss.item()) * n
            seen += n

            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                type=f"{type_loss.item():.3f}",
                seg=f"{segment_loss.item():.3f}",
            )

        train_loss /= max(1, seen)
        metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"type_acc={metrics['type_acc']*100:6.2f}% "
            f"char_acc={metrics['char_acc']*100:6.2f}% "
            f"oracle_char={metrics['oracle_type_char_acc']*100:6.2f}% "
            f"exact_bits={metrics['exact_bits_acc']*100:6.2f}% "
            f"segment_acc={metrics['segment_acc']*100:6.2f}%"
        )

        # End-to-end character accuracy is primary when a configured charset
        # is available. For type-only datasets, fall back to type accuracy.
        score = (
            metrics["char_acc"]
            if metrics["char_samples"] > 0
            else metrics["type_acc"]
        )

        if score > best_val:
            best_val = score
            best_epoch = epoch
            best_metrics = metrics
            stale = 0

            checkpoint = {
                "architecture": "segment-multitask-v2",
                "model_state": model.state_dict(),
                "model_id": model_id,
                "created_at": created_at,
                "best_epoch": best_epoch,
                "val_metrics": best_metrics,
                "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
                "segment_threshold": SEGMENT_THRESHOLD,
                "decoder": "probabilistic-v2",
                "supported_segment_types": list(SEGMENT_TYPES),
                "train_source_images": train_sources,
                "validation_source_images": val_sources,
            }
            torch.save(checkpoint, checkpoint_path)

            register_model(
                model_id,
                checkpoint_path,
                {
                    "architecture": "segment-multitask-v2",
                    "created_at": created_at,
                    "best_epoch": best_epoch,
                    "val_metrics": best_metrics,
                    "train_source_images": train_sources,
                    "validation_source_images": val_sources,
                    "decoder": "probabilistic-v2",
                    "supported_segment_types": list(SEGMENT_TYPES),
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
        print(
            "Best validation character accuracy: "
            f"{best_metrics['char_acc']*100:.2f}%"
        )


if __name__ == "__main__":
    main()
