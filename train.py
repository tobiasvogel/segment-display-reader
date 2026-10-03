from pathlib import Path
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
)
from src.dataset import collect_samples, SegmentDataset
from src.transforms import train_transform, eval_transform
from src.split import split_samples
from src.model import SegmentCNN
from src.labels import decode_segments


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_samples = 0
    char_correct = 0
    exact_bits_correct = 0
    segment_correct = 0
    segment_total = 0
    loss_sum = 0.0

    criterion = nn.BCEWithLogitsLoss()

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)
        bits = (probs >= SEGMENT_THRESHOLD).int()

        loss_sum += float(loss.item()) * images.size(0)
        total_samples += images.size(0)

        target_bits = targets.int()
        exact_bits_correct += int(
            torch.all(bits == target_bits, dim=1).sum().item()
        )

        segment_correct += int((bits == target_bits).sum().item())
        segment_total += int(target_bits.numel())

        for pred_bits, true_label in zip(bits.cpu().tolist(), batch["label"]):
            pred_char = decode_segments(pred_bits)
            char_correct += int(pred_char == true_label)

    return {
        "loss": loss_sum / max(1, total_samples),
        "char_acc": char_correct / max(1, total_samples),
        "exact_bits_acc": exact_bits_correct / max(1, total_samples),
        "segment_acc": segment_correct / max(1, segment_total),
    }


def main():
    seed_everything(RANDOM_SEED)

    # WICHTIG:
    # Der Split wird über Original- und Augmentierungsdateien gemeinsam gemacht.
    # Für echte Experimente sollten augmentierte Varianten eines Originals nicht
    # über Train/Val verteilt werden. Der Display-ID-Split vermeidet das sauber.
    samples = collect_samples(RAW_DIR, AUGMENTED_DIR)

    if len(samples) < 2:
        raise RuntimeError("Zu wenige Samples zum Trainieren.")

    train_samples, val_samples = split_samples(samples)

    print(f"Train: {len(train_samples)}")
    print(f"Validation: {len(val_samples)}")

    train_ds = SegmentDataset(train_samples, train_transform())
    val_ds = SegmentDataset(val_samples, eval_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
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
    criterion = nn.BCEWithLogitsLoss()

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_model.pt"

    best_val = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss = 0.0
        seen = 0

        bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{EPOCHS}",
            leave=False
        )

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
            f"Epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"char_acc={metrics['char_acc']*100:6.2f}% "
            f"exact_bits={metrics['exact_bits_acc']*100:6.2f}% "
            f"segment_acc={metrics['segment_acc']*100:6.2f}%"
        )

        # Primäres Kriterium: vollständige Ziffer korrekt.
        score = metrics["char_acc"]

        if score > best_val:
            best_val = score
            epochs_without_improvement = 0

            torch.save({
                "model_state": model.state_dict(),
                "val_metrics": metrics,
                "epoch": epoch,
            }, best_path)

            print(f"  -> neues bestes Modell: {best_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping.")
            break

    print(f"\nBestes Modell: {best_path}")


if __name__ == "__main__":
    main()
