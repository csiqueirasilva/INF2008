import os
import json
from pathlib import Path
import random
import click
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .root import cli
from ..core.models.unet import UNet
from ..core.drr_utils import device
from ..core.overlay import save_u8_gray


class ImageMaskDataset(Dataset):
    def __init__(self, manifest_csv: Path, split: str):
        import pandas as pd
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _augment(self, img, msk):
        # flips
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            msk = cv2.flip(msk, 1)
        # jitter
        if random.random() < 0.5:
            alpha = 1.0 + (random.random() - 0.5) * 0.4  # contrast
            beta = (random.random() - 0.5) * 40          # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        # blur
        if random.random() < 0.3:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        # noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img, msk

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row["image"]), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(row["mask"]), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise FileNotFoundError(row["image"] if img is None else row["mask"])
        # ensure same size
        if img.shape[:2] != msk.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        # binarize
        msk = ((msk > 0).astype(np.float32))
        img, msk = self._augment(img, msk)
        # to tensors
        x = torch.from_numpy((img / 255.0).astype(np.float32))[None, ...]
        y = torch.from_numpy(msk.astype(np.float32))[None, ...]
        return x, y, str(row["image"])  # include path for debugging


def dice_coeff(pred, target, eps=1e-6):
    # pred, target: N,1,H,W in [0,1]
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dice = 1.0 - (2 * (probs * target).sum() + 1e-6) / (probs.sum() + target.sum() + 1e-6)
        return bce + dice


@cli.command("train-seg2d")
@click.option("--dataset-dir", type=click.Path(path_type=Path), required=True, help="Folder with images/, masks/, manifest.csv")
@click.option("--epochs", type=int, default=20, show_default=True)
@click.option("--batch", type=int, default=4, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("outputs/seg2d"), show_default=True)
def train_seg2d(dataset_dir, epochs, batch, lr, out_dir):
    """Train a simple 2D UNet on synthetic DRR dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = device()

    manifest = Path(dataset_dir) / "manifest.csv"
    if not manifest.exists():
        raise click.ClickException(f"manifest.csv not found in {dataset_dir}")

    ds_tr = ImageMaskDataset(manifest, split="train")
    ds_va = ImageMaskDataset(manifest, split="val")
    if len(ds_tr) == 0 or len(ds_va) == 0:
        raise click.ClickException("Dataset has empty train/val split. Increase limit or val-ratio.")
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(in_ch=1, out_ch=1, base=32).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = DiceBCELoss()

    best = {"dice": -1.0}
    for ep in range(1, epochs + 1):
        model.train(); tr_loss = 0.0
        for x, y, _ in dl_tr:
            x = x.to(dev); y = y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward(); opt.step()
            tr_loss += float(loss.detach().cpu().item()) * x.size(0)
        tr_loss /= len(ds_tr)

        # validate
        model.eval(); dices = []
        with torch.no_grad():
            for x, y, _ in dl_va:
                x = x.to(dev); y = y.to(dev)
                logits = model(x)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                dices.append(dice_coeff(preds, y))
        va_dice = float(np.mean(dices)) if dices else 0.0

        click.echo(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_dice={va_dice:.4f}")

        # save best
        if va_dice > best["dice"]:
            best.update({"dice": va_dice, "epoch": ep})
            ckpt = out_dir / "model.pt"
            torch.save({"model": model.state_dict(), "meta": best}, ckpt)

            # write a few previews
            prev_dir = out_dir / "val_previews"; prev_dir.mkdir(exist_ok=True)
            n_prev = 6
            it = iter(dl_va)
            with torch.no_grad():
                for i in range(n_prev):
                    try:
                        x, y, paths = next(it)
                    except StopIteration:
                        break
                    x = x.to(dev); y = y.to(dev)
                    logits = model(x)
                    probs = torch.sigmoid(logits)
                    for b in range(x.size(0)):
                        img = (x[b, 0].detach().cpu().numpy() * 255).astype(np.uint8)
                        gt = (y[b, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
                        pr = (probs[b, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
                        base = Path(paths[b]).stem
                        save_u8_gray(prev_dir / f"{base}_img.png", img)
                        save_u8_gray(prev_dir / f"{base}_gt.png", gt)
                        save_u8_gray(prev_dir / f"{base}_pred.png", pr)

    with open(out_dir / "best.json", "w") as f:
        json.dump(best, f, indent=2)
    click.echo(f"✅ Training complete. Best dice={best['dice']:.4f} at epoch {best['epoch']}")

