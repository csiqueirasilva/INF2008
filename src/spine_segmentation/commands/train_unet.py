from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import click
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .root import cli


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SpineDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], mask_paths: Sequence[Path]):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        max_label = 0
        for mask_path in self.mask_paths:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise FileNotFoundError(f"Failed to read mask: {mask_path}")
            max_label = max(max_label, int(mask.max()))
        self.num_classes = max_label + 1  # include background

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        mask = mask.astype(np.int64)

        return torch.from_numpy(img), torch.from_numpy(mask)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                     diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, base: int = 32):
        super().__init__()
        self.inc = DoubleConv(n_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 8)
        self.up1 = Up(base * 16, base * 4)
        self.up2 = Up(base * 8, base * 2)
        self.up3 = Up(base * 4, base)
        self.up4 = Up(base * 2, base)
        self.outc = OutConv(base, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


@dataclass
class TrainConfig:
    csv_path: Path
    out_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    val_fraction: float
    num_workers: int
    device: str
    seed: int


def _load_manifest(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"image", "mask_labels"}
    missing = expected - set(df.columns)
    if missing:
        raise click.ClickException(f"Manifest {csv_path} missing columns: {missing}")
    return df


def _split_dataset(dataset: SpineDataset, val_fraction: float, seed: int):
    if val_fraction <= 0:
        return dataset, None
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    return train_set, val_set


def _save_metadata(config: TrainConfig, num_classes: int, history: List[dict], best_path: Path) -> None:
    meta = {
        "csv": str(config.csv_path),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "val_fraction": config.val_fraction,
        "num_workers": config.num_workers,
        "device": config.device,
        "seed": config.seed,
        "num_classes": num_classes,
        "best_model": str(best_path),
        "history": history,
    }
    with open(config.out_dir / "training_summary.json", "w") as fh:
        json.dump(meta, fh, indent=2)


@cli.command("train-unet")
@click.option("--csv", "csv_path", type=click.Path(path_type=Path), required=True,
              help="Path to manifest.csv (must contain 'image' and 'mask_labels' columns)")
@click.option("--out-dir", type=click.Path(path_type=Path), required=True,
              help="Directory to store checkpoints and logs")
@click.option("--epochs", type=int, default=5, show_default=True)
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--learning-rate", type=float, default=1e-3, show_default=True)
@click.option("--val-fraction", type=float, default=0.1, show_default=True,
              help="Fraction of samples to use for validation (0 disables validation)")
@click.option("--num-workers", type=int, default=2, show_default=True)
@click.option("--device", default=None, help="Device to train on (default: cuda if available else cpu)")
@click.option("--seed", type=int, default=42, show_default=True)
def train_unet(csv_path: Path, out_dir: Path, epochs: int, batch_size: int, learning_rate: float,
              val_fraction: float, num_workers: int, device: str | None, seed: int) -> None:
    """Train a simple UNet using the manifest generated by build_pseudo_dataset.sh."""

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(seed)

    df = _load_manifest(csv_path)
    image_paths = [Path(p).expanduser().resolve() for p in df["image"].tolist()]
    mask_paths = [Path(p).expanduser().resolve() for p in df["mask_labels"].tolist()]

    dataset = SpineDataset(image_paths, mask_paths)
    num_classes = dataset.num_classes

    train_set, val_set = _split_dataset(dataset, val_fraction, seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TrainConfig(csv_path=csv_path.expanduser().resolve(), out_dir=out_dir,
                         epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                         val_fraction=val_fraction, num_workers=num_workers,
                         device=device, seed=seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

    model = UNet(n_channels=1, n_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history: List[dict] = []
    best_val = float("inf")
    best_path = out_dir / "unet_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_loss = None
        if val_loader is not None:
            model.eval()
            running_val = 0.0
            with torch.no_grad():
                for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                    running_val += loss.item() * imgs.size(0)
            val_loss = running_val / len(val_loader.dataset)

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "config": config.__dict__,
                }, best_path)
        else:
            # No validation: always keep the last model as best
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "config": config.__dict__,
            }, best_path)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        click.echo(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f}" +
                   (f" val_loss={val_loss:.4f}" if val_loss is not None else ""))

    _save_metadata(config, num_classes, history, best_path)
    click.echo(f"✅ Training complete. Best model saved to {best_path}")
