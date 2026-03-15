"""
Training loop for AlphaNet.
Training is time-constrained: stops after TIME_LIMIT seconds (default 5 minutes).

Usage:
    python train.py
"""

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ChessDataset
from model import AlphaNet

DATASET = "data"
BATCH_SIZE = 256
LR = 1e-3
VALUE_LOSS_WEIGHT = 0.1
SAVE_PATH = "model.pt"
TIME_LIMIT = 15 * 60  # 5 minutes


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Time limit: {TIME_LIMIT}s")

    model = AlphaNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_ds = ChessDataset(DATASET, split="train")
    val_ds = ChessDataset(DATASET, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_ds)} positions, Val: {len(val_ds)} positions")

    start_time = time.time()
    epoch = 0
    time_up = False

    while not time_up:
        epoch += 1

        # Train
        model.train()
        total_loss, total_pacc, n_batches = 0.0, 0.0, 0

        for x, (y_policy, y_value) in train_loader:
            if time.time() - start_time >= TIME_LIMIT:
                time_up = True
                break

            x = x.to(device)
            y_policy = y_policy.to(device)
            y_value = y_value.to(device)

            pred_policy, pred_value = model(x)
            policy_loss = F.cross_entropy(pred_policy, y_policy)
            value_loss = F.mse_loss(pred_value, y_value)
            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pacc += (pred_policy.argmax(1) == y_policy).float().mean().item()
            n_batches += 1

        if n_batches == 0:
            break

        avg_loss = total_loss / n_batches
        avg_pacc = total_pacc / n_batches
        elapsed = time.time() - start_time

        # Validate
        model.eval()
        val_loss, val_pacc, val_n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, (y_policy, y_value) in val_loader:
                x = x.to(device)
                y_policy = y_policy.to(device)
                y_value = y_value.to(device)

                pred_policy, pred_value = model(x)
                policy_loss = F.cross_entropy(pred_policy, y_policy)
                value_loss = F.mse_loss(pred_value, y_value)

                val_loss += (policy_loss + VALUE_LOSS_WEIGHT * value_loss).item()
                val_pacc += (pred_policy.argmax(1) == y_policy).float().mean().item()
                val_n += 1

        val_avg_loss = val_loss / max(val_n, 1)
        val_avg_pacc = val_pacc / max(val_n, 1)

        print(
            f"Epoch {epoch} [{elapsed:.0f}s/{TIME_LIMIT}s]  "
            f"loss={avg_loss:.4f} pacc={avg_pacc:.3f}  "
            f"val_loss={val_avg_loss:.4f} val_pacc={val_avg_pacc:.3f}"
        )

    elapsed = time.time() - start_time
    print("---")
    print(f"Training finished: {epoch} epochs in {elapsed:.1f}s")
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"peak_vram_mb: {peak_mb:.1f}")
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved weights to {SAVE_PATH}")


if __name__ == "__main__":
    train()
