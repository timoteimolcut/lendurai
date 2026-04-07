# AI-generated
"""
train.py — Training loop + single-datapoint overfit test.

Usage
-----
# Full training run (synthetic data):
    python train.py

# Overfit test (single pair, should reach loss < 0.01 in ~200 steps):
    python train.py --overfit

# Resume from checkpoint:
    python train.py --checkpoint checkpoints/epoch_5.pt

Options
-------
--overfit           Run single-datapoint overfit sanity check.
--epochs N          Number of training epochs (default 20).
--batch-size N      Batch size (default 32).
--lr LR             Learning rate (default 3e-4).
--temperature T     InfoNCE temperature (default 0.07).
--embedding-dim D   Embedding dimensionality (default 128).
--checkpoint PATH   Path to a .pt checkpoint to resume from.
--save-dir DIR      Directory to save checkpoints (default checkpoints/).
--device DEVICE     torch device string, e.g. cpu / cuda / mps.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from models.encoder import Encoder
from data.dataset import SyntheticPairDataset
from losses.infonce import InfoNCELoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, epoch: int, model: Encoder, optimizer, loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(path: Path, model: Encoder, optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resumed from {path} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    return start_epoch


# ---------------------------------------------------------------------------
# Overfit test
# ---------------------------------------------------------------------------

def run_overfit_test(args: argparse.Namespace) -> None:
    """
    Train on a *single* positive pair for many steps.
    The loss must converge to near zero, proving that:
      1. Gradients flow through the encoder and head.
      2. InfoNCE collapses correctly when batch_size==2 (one pos, one neg).

    We use batch_size=2: the single pair repeated twice (both directions).
    After ~200 gradient steps the loss should be < 0.01.
    """
    print("=" * 60)
    print("OVERFIT TEST — single synthetic data point")
    print("=" * 60)

    device = get_device(args.device)
    print(f"Device: {device}")

    # One fixed pair — index 0 repeated
    dataset = SyntheticPairDataset(num_samples=2, seed=0)   # need >= 2 for InfoNCE
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    drone_view, sat_view, _ = next(iter(loader))
    drone_view = drone_view.to(device)
    sat_view = sat_view.to(device)

    model = Encoder(embedding_dim=args.embedding_dim, pretrained=False).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = InfoNCELoss(temperature=args.temperature)

    target_loss = 0.05
    max_steps = 500
    print(f"Optimising for up to {max_steps} steps (target loss < {target_loss})…\n")

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        q = model(drone_view)
        k = model(sat_view)
        loss = criterion(q, k)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"  step {step:4d} | loss = {loss.item():.6f}")

        if loss.item() < target_loss:
            print(f"\n✓ Overfit test PASSED at step {step} (loss={loss.item():.6f})")
            return

    print(f"\n✗ Overfit test FAILED — loss did not reach {target_loss} in {max_steps} steps.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    print(f"Device: {device}")

    dataset = SyntheticPairDataset(num_samples=args.dataset_size, seed=42)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
    )

    model = Encoder(embedding_dim=args.embedding_dim, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = InfoNCELoss(temperature=args.temperature)

    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(Path(args.checkpoint), model, optimizer)

    save_dir = Path(args.save_dir)

    print(f"\nTraining for {args.epochs} epochs, batch_size={args.batch_size}\n")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        num_batches = 0
        for drone_view, sat_view, _ in loader:
            drone_view = drone_view.to(device, non_blocking=True)
            sat_view = sat_view.to(device, non_blocking=True)

            optimizer.zero_grad()
            q = model(drone_view)
            k = model(sat_view)
            loss = criterion(q, k)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * drone_view.size(0)
            num_batches += drone_view.size(0)

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{args.epochs} | loss={avg_loss:.4f} | {elapsed:.1f}s")

        if (epoch + 1) % 5 == 0:
            ckpt_path = save_dir / f"epoch_{epoch+1:03d}.pt"
            save_checkpoint(ckpt_path, epoch, model, optimizer, avg_loss)

    # Save final checkpoint
    save_checkpoint(save_dir / "final.pt", args.epochs - 1, model, optimizer, avg_loss)
    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the drone localisation encoder.")
    p.add_argument("--overfit", action="store_true", help="Run single-datapoint overfit test.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.overfit:
        run_overfit_test(args)
    else:
        run_training(args)
