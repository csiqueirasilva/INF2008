from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import click
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .root import cli

try:  # torch>=2.0 recommended API
    from torch import amp as _torch_amp

    _AUTOMATIC_CAST_IMPL = _torch_amp.autocast
    GradScaler = _torch_amp.GradScaler
    _AUTOCAST_NEEDS_DEVICE = True
except (ImportError, AttributeError):  # fallback for older torch where torch.amp is missing
    from torch.cuda.amp import autocast as _cuda_autocast  # type: ignore[attr-defined]
    from torch.cuda.amp import GradScaler  # type: ignore[attr-defined]

    _AUTOMATIC_CAST_IMPL = _cuda_autocast
    _AUTOCAST_NEEDS_DEVICE = False


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SpineDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        mask_paths: Sequence[Path],
        transform: Optional[Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = None,
        num_classes: Optional[int] = None,
        use_otsu_channel: bool = False,
        use_coord_channels: bool = False,
        otsu_blur_kernel: int = 5,
    ):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.transform = transform
        self.use_otsu_channel = use_otsu_channel
        self.use_coord_channels = use_coord_channels
        if otsu_blur_kernel <= 0:
            otsu_blur_kernel = 1
        if otsu_blur_kernel % 2 == 0:
            otsu_blur_kernel += 1
        self.otsu_blur_kernel = otsu_blur_kernel
        if num_classes is None:
            max_label = 0
            for mask_path in self.mask_paths:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise FileNotFoundError(f"Failed to read mask: {mask_path}")
                max_label = max(max_label, int(mask.max()))
            self.num_classes = max_label + 1  # include background
        else:
            self.num_classes = num_classes

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

        img_f = img.astype(np.float32) / 255.0
        channels = [img_f]

        if self.use_otsu_channel:
            blur = cv2.GaussianBlur(img, (self.otsu_blur_kernel, self.otsu_blur_kernel), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            channels.append((otsu.astype(np.float32) / 255.0))

        if self.use_coord_channels:
            h, w = img.shape
            ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
            xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            channels.extend([yy, xx])

        img_stacked = np.stack(channels, axis=-1)  # (H, W, C)
        mask = mask.astype(np.int64)

        if self.transform is not None:
            img_stacked, mask = self.transform(img_stacked, mask)

        img = np.transpose(img_stacked, (2, 0, 1))  # (C, H, W)

        return torch.from_numpy(img), torch.from_numpy(mask)


def _augment_pair(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply simple spatial and intensity augmentations in-place safe manner."""

    augmented_img = img.copy()  # (H, W, C)
    augmented_mask = mask.copy()

    # Horizontal flip (mirror) keeps anatomy plausible for lateral views
    if random.random() < 0.5:
        augmented_img = np.fliplr(augmented_img)
        augmented_mask = np.fliplr(augmented_mask)

    # Small rotation with reflection padding, preserves label discreteness
    if random.random() < 0.3:
        angle = random.uniform(-10.0, 10.0)
        h, w, _ = augmented_img.shape
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        augmented_img = cv2.warpAffine(
            augmented_img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        augmented_mask = cv2.warpAffine(
            augmented_mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    # Mild intensity scaling and bias on the first channel (assumed clahe/base image)
    if random.random() < 0.4:
        scale = random.uniform(0.9, 1.1)
        bias = random.uniform(-0.05, 0.05)
        augmented_img[..., 0] = np.clip(augmented_img[..., 0] * scale + bias, 0.0, 1.0)

    # Low magnitude Gaussian noise keeps denoising behaviour realistic (first channel only)
    if random.random() < 0.3:
        noise = np.random.normal(0.0, 0.015, size=augmented_img[..., 0].shape).astype(np.float32)
        augmented_img[..., 0] = np.clip(augmented_img[..., 0] + noise, 0.0, 1.0)

    return augmented_img.astype(np.float32), augmented_mask.astype(np.int64)


class MetricTracker:
    """Collects confusion statistics and derives segmentation metrics."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
        self.loss_sum = 0.0
        self.sample_count = 0
        self.correct = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor, batch_size: int) -> None:
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            flat_targets = targets.view(-1).to(torch.long)
            flat_preds = preds.view(-1).to(torch.long)
            cm = torch.bincount(
                flat_targets * self.num_classes + flat_preds,
                minlength=self.num_classes ** 2,
            ).reshape(self.num_classes, self.num_classes)
            cm = cm.cpu().numpy()

        self.confusion += cm
        self.correct += float(np.trace(cm))
        if isinstance(loss, torch.Tensor):
            loss_value = float(loss.detach().item())
        else:
            loss_value = float(loss)
        self.loss_sum += loss_value * batch_size
        self.sample_count += batch_size

    def compute(self) -> Dict[str, object]:
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp
        support = tp + fn
        pred_support = tp + fp
        union = tp + fp + fn

        precision = np.divide(tp, pred_support, out=np.zeros_like(tp), where=pred_support > 0)
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(tp), where=(precision + recall) > 0)
        dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
        iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)

        valid_support = support > 0
        valid_union = union > 0
        avg_loss = self.loss_sum / max(1, self.sample_count)
        total = self.confusion.sum()
        accuracy = float(tp.sum() / total) if total > 0 else 0.0

        macro_precision = float(precision[valid_support].mean()) if valid_support.any() else 0.0
        macro_recall = float(recall[valid_support].mean()) if valid_support.any() else 0.0
        macro_f1 = float(f1[valid_support].mean()) if valid_support.any() else 0.0
        macro_dice = float(dice[valid_union].mean()) if valid_union.any() else 0.0
        macro_iou = float(iou[valid_union].mean()) if valid_union.any() else 0.0

        per_class: Dict[str, Dict[str, float]] = {}
        for idx in range(self.num_classes):
            per_class[str(idx)] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "dice": float(dice[idx]),
                "iou": float(iou[idx]),
                "support": float(support[idx]),
            }

        return {
            "loss": float(avg_loss),
            "accuracy": accuracy,
            "macro": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
                "dice": macro_dice,
                "iou": macro_iou,
            },
            "per_class": per_class,
        }


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if _AUTOCAST_NEEDS_DEVICE:
        return _AUTOMATIC_CAST_IMPL(device.type, enabled=enabled)
    return _AUTOMATIC_CAST_IMPL(enabled=enabled)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: Optional[optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    amp_enabled: bool = False,
    clip_grad_norm: Optional[float] = None,
    desc: str = "",
) -> Dict[str, object]:
    tracker = MetricTracker(num_classes)
    is_train = optimizer is not None
    model.train(mode=is_train)

    iterator = tqdm(loader, desc=desc, leave=False)

    grad_context = nullcontext() if is_train else torch.no_grad()
    with grad_context:
        for imgs, masks in iterator:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            batch_size = imgs.size(0)

            with _autocast_context(device, amp_enabled):
                logits = model(imgs)
                loss = criterion(logits, masks)

            if is_train:
                assert optimizer is not None and scaler is not None
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if clip_grad_norm is not None and clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                loss_for_metrics = loss.detach()
            else:
                loss_for_metrics = loss.detach()

            tracker.update(logits.detach(), masks, loss_for_metrics, batch_size)

    return tracker.compute()


def _select_metric_value(metrics: Dict[str, object], monitor: str) -> float:
    if monitor == "loss":
        return float(metrics["loss"])
    macro = metrics.get("macro", {})
    value = macro.get(monitor)
    if value is None:
        raise KeyError(f"Metric '{monitor}' not present in metrics dictionary")
    return float(value)


def _flatten_metrics_row(
    epoch: int,
    split: str,
    metrics: Dict[str, object],
    lr: Optional[float] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "epoch": epoch,
        "split": split,
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["macro"]["precision"]),
        "recall": float(metrics["macro"]["recall"]),
        "f1": float(metrics["macro"]["f1"]),
        "dice": float(metrics["macro"]["dice"]),
        "iou": float(metrics["macro"]["iou"]),
    }
    if lr is not None:
        row["lr"] = float(lr)
    return row


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
        self.down4 = Down(base * 8, base * 16)
        self.up1 = Up(base * 16, base * 8)
        self.up2 = Up(base * 8, base * 4)
        self.up3 = Up(base * 4, base * 2)
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
    test_fraction: float
    num_workers: int
    device: str
    seed: int
    patience: int
    min_delta: float
    monitor: str
    clip_grad_norm: Optional[float]
    amp: bool
    augment: bool
    lr_factor: float
    lr_patience: int
    lr_min: float
    log_dir: Optional[Path]
    use_otsu_channel: bool
    use_coord_channels: bool
    otsu_blur_kernel: int
    num_input_channels: int


def _load_manifest(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"image", "mask_labels"}
    missing = expected - set(df.columns)
    if missing:
        raise click.ClickException(f"Manifest {csv_path} missing columns: {missing}")
    return df


def _split_dataset(
    image_paths: Sequence[Path],
    mask_paths: Sequence[Path],
    val_fraction: float,
    test_fraction: float,
    seed: int,
):
    if val_fraction < 0 or test_fraction < 0:
        raise click.ClickException("val_fraction and test_fraction must be non-negative")
    if val_fraction + test_fraction >= 1:
        raise click.ClickException("val_fraction + test_fraction must be < 1")

    total = len(image_paths)
    if total == 0:
        raise click.ClickException("Dataset is empty")

    val_size = int(total * val_fraction)
    test_size = int(total * test_fraction)
    if val_fraction > 0 and val_size == 0:
        val_size = 1
    if test_fraction > 0 and test_size == 0:
        test_size = 1

    train_size = total - val_size - test_size
    if train_size <= 0:
        raise click.ClickException(
            "Not enough samples for the requested val/test fractions; reduce the split sizes"
        )

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_indices = indices[:val_size]
    test_indices = indices[val_size:val_size + test_size]
    train_indices = indices[val_size + test_size:]

    def _gather(idxs: Sequence[int]) -> tuple[List[Path], List[Path]]:
        return [image_paths[i] for i in idxs], [mask_paths[i] for i in idxs]

    train_paths = _gather(train_indices)
    val_paths = _gather(val_indices) if val_size > 0 else None
    test_paths = _gather(test_indices) if test_size > 0 else None

    return train_paths, val_paths, test_paths


def _save_metadata(
    config: TrainConfig,
    num_classes: int,
    history: List[dict],
    best_path: Path,
    best_epoch: Optional[int],
    best_metrics: Optional[dict],
    test_metrics: Optional[dict],
) -> None:
    meta = {
        "csv": str(config.csv_path),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "val_fraction": config.val_fraction,
        "test_fraction": config.test_fraction,
        "num_workers": config.num_workers,
        "device": config.device,
        "seed": config.seed,
        "patience": config.patience,
        "min_delta": config.min_delta,
        "monitor": config.monitor,
        "clip_grad_norm": config.clip_grad_norm,
        "amp": config.amp,
        "augment": config.augment,
        "lr_factor": config.lr_factor,
        "lr_patience": config.lr_patience,
        "lr_min": config.lr_min,
        "log_dir": str(config.log_dir) if config.log_dir is not None else None,
        "num_classes": num_classes,
        "num_input_channels": config.num_input_channels,
        "use_otsu_channel": config.use_otsu_channel,
        "use_coord_channels": config.use_coord_channels,
        "otsu_blur_kernel": config.otsu_blur_kernel,
        "best_model": str(best_path),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }
    with open(config.out_dir / "training_summary.json", "w") as fh:
        json.dump(meta, fh, indent=2)


def _load_checkpoint(path: Path, map_location: torch.device) -> dict:
    """Load a checkpoint while remaining compatible with torch>=2.6 weights_only default."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch does not accept weights_only – fall back to the legacy behaviour.
        return torch.load(path, map_location=map_location)


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
@click.option("--test-fraction", type=float, default=0.1, show_default=True,
              help="Fraction of samples to reserve for testing (evaluated after training)")
@click.option("--num-workers", type=int, default=2, show_default=True)
@click.option("--device", default=None, help="Device to train on (default: cuda if available else cpu)")
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--patience", type=int, default=5, show_default=True,
              help="Number of epochs with no improvement before early stopping")
@click.option("--min-delta", type=float, default=0.0, show_default=True,
              help="Minimum change in monitored metric to qualify as improvement")
@click.option("--monitor", type=click.Choice(["dice", "iou", "f1", "loss"]), default="dice",
              show_default=True, help="Metric to monitor for early stopping and checkpointing")
@click.option("--clip-grad-norm", type=float, default=1.0, show_default=True,
              help="Gradient norm clipping value (<=0 disables clipping)")
@click.option("--amp/--no-amp", default=True, show_default=True,
              help="Enable automatic mixed precision on CUDA")
@click.option("--augment/--no-augment", default=True, show_default=True,
              help="Apply data augmentations to the training split")
@click.option("--lr-factor", type=float, default=0.5, show_default=True,
              help="Multiplicative factor for ReduceLROnPlateau scheduler")
@click.option("--lr-patience", type=int, default=2, show_default=True,
              help="Epochs with no improvement before reducing LR")
@click.option("--lr-min", type=float, default=1e-6, show_default=True,
              help="Lower bound for the learning rate scheduler")
@click.option("--log-dir", type=click.Path(path_type=Path), default=None,
              help="Optional directory for TensorBoard event files (defaults to out_dir/tensorboard)")
@click.option("--use-otsu-channel/--no-otsu-channel", default=False, show_default=True,
              help="Append an Otsu foreground channel (computed with --otsu-blur-kernel) to the input.")
@click.option("--use-coord-channels/--no-coord-channels", default=False, show_default=True,
              help="Append normalized y/x coordinate channels to the input.")
@click.option("--otsu-blur-kernel", type=int, default=5, show_default=True,
              help="Gaussian blur kernel before Otsu when --use-otsu-channel is enabled (odd, >=1).")
def train_unet(csv_path: Path, out_dir: Path, epochs: int, batch_size: int, learning_rate: float,
              val_fraction: float, test_fraction: float, num_workers: int, device: str | None,
              seed: int, patience: int, min_delta: float, monitor: str, clip_grad_norm: float,
              amp: bool, augment: bool, lr_factor: float, lr_patience: int, lr_min: float,
              log_dir: Path | None, use_otsu_channel: bool, use_coord_channels: bool,
              otsu_blur_kernel: int) -> None:
    """Train a simple UNet using the manifest generated by build_pseudo_dataset.sh."""

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(seed)

    df = _load_manifest(csv_path)
    image_paths = [Path(p).expanduser().resolve() for p in df["image"].tolist()]
    mask_paths = [Path(p).expanduser().resolve() for p in df["mask_labels"].tolist()]

    dataset = SpineDataset(
        image_paths,
        mask_paths,
        use_otsu_channel=use_otsu_channel,
        use_coord_channels=use_coord_channels,
        otsu_blur_kernel=otsu_blur_kernel,
    )
    num_classes = dataset.num_classes

    train_paths, val_paths, test_paths = _split_dataset(
        dataset.image_paths, dataset.mask_paths, val_fraction, test_fraction, seed
    )

    num_input_channels = 1 + (1 if use_otsu_channel else 0) + (2 if use_coord_channels else 0)

    train_dataset = SpineDataset(
        train_paths[0],
        train_paths[1],
        transform=_augment_pair if augment else None,
        num_classes=num_classes,
        use_otsu_channel=use_otsu_channel,
        use_coord_channels=use_coord_channels,
        otsu_blur_kernel=otsu_blur_kernel,
    )
    val_dataset = (
        SpineDataset(
            val_paths[0],
            val_paths[1],
            num_classes=num_classes,
            use_otsu_channel=use_otsu_channel,
            use_coord_channels=use_coord_channels,
            otsu_blur_kernel=otsu_blur_kernel,
        )
        if val_paths is not None
        else None
    )
    test_dataset = (
        SpineDataset(
            test_paths[0],
            test_paths[1],
            num_classes=num_classes,
            use_otsu_channel=use_otsu_channel,
            use_coord_channels=use_coord_channels,
            otsu_blur_kernel=otsu_blur_kernel,
        )
        if test_paths is not None
        else None
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        torch_device = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(f"Invalid device '{device}': {exc}") from exc

    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("CUDA was requested but is not available on this machine")

    pin_memory = torch_device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        if val_dataset is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        if test_dataset is not None
        else None
    )

    amp_enabled = amp and torch_device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    model = UNet(n_channels=num_input_channels, n_classes=num_classes).to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=lr_factor,
        patience=lr_patience,
        min_lr=lr_min,
    )

    clip_value = clip_grad_norm if clip_grad_norm > 0 else None

    log_dir_path: Optional[Path]
    writer = None
    if log_dir is None:
        log_dir_path = out_dir / "tensorboard"
    else:
        log_dir_path = log_dir.expanduser().resolve()

    if log_dir_path is not None:
        log_dir_path.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(log_dir_path))
        except ImportError:
            click.echo("⚠️ TensorBoard is not installed; skipping event logging.", err=True)
            writer = None
            log_dir_path = None

    config = TrainConfig(
        csv_path=csv_path.expanduser().resolve(),
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        num_workers=num_workers,
        device=str(torch_device),
        seed=seed,
        patience=patience,
        min_delta=min_delta,
        monitor=monitor,
        clip_grad_norm=clip_value,
        amp=amp_enabled,
        augment=augment,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
        lr_min=lr_min,
        log_dir=log_dir_path,
        use_otsu_channel=use_otsu_channel,
        use_coord_channels=use_coord_channels,
        otsu_blur_kernel=otsu_blur_kernel,
        num_input_channels=num_input_channels,
    )

    history: List[dict] = []
    metrics_rows: List[Dict[str, object]] = []
    best_path = out_dir / "unet_best.pt"
    best_epoch: Optional[int] = None
    best_metric = math.inf if monitor == "loss" else -math.inf
    best_metrics: Optional[dict] = None
    patience_counter = 0
    current_lr = optimizer.param_groups[0]["lr"]

    click.echo(
        f"Starting training for {epochs} epochs on {torch_device} (AMP={'on' if amp_enabled else 'off'})"
    )

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            torch_device,
            num_classes,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            clip_grad_norm=clip_value,
            desc=f"Epoch {epoch}/{epochs} [train]",
        )

        val_metrics = None
        if val_loader is not None:
            val_metrics = _run_epoch(
                model,
                val_loader,
                criterion,
                torch_device,
                num_classes,
                optimizer=None,
                scaler=None,
                amp_enabled=False,
                clip_grad_norm=None,
                desc=f"Epoch {epoch}/{epochs} [val]",
            )
            scheduler_metric = val_metrics["loss"]
        else:
            scheduler_metric = train_metrics["loss"]

        scheduler.step(scheduler_metric)

        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": current_lr,
            }
        )

        metrics_rows.append(_flatten_metrics_row(epoch, "train", train_metrics, current_lr))
        if val_metrics is not None:
            metrics_rows.append(_flatten_metrics_row(epoch, "val", val_metrics, current_lr))

        if writer is not None:
            writer.add_scalar("loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("dice/train", train_metrics["macro"]["dice"], epoch)
            if val_metrics is not None:
                writer.add_scalar("loss/val", val_metrics["loss"], epoch)
                writer.add_scalar("dice/val", val_metrics["macro"]["dice"], epoch)

        monitor_metrics = val_metrics if val_metrics is not None else train_metrics
        metric_value = _select_metric_value(monitor_metrics, monitor)
        improved = (
            metric_value < (best_metric - min_delta)
            if monitor == "loss"
            else metric_value > (best_metric + min_delta)
        )

        if improved:
            best_metric = metric_value
            best_epoch = epoch
            best_metrics = monitor_metrics
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "config": config.__dict__,
                },
                best_path,
            )
        else:
            patience_counter += 1

        click.echo(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['macro']['dice']:.4f}"
            + (
                f" | val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['macro']['dice']:.4f}"
                if val_metrics is not None
                else ""
            )
            + f" | lr={current_lr:.2e}"
        )

        if patience > 0 and patience_counter >= patience:
            click.echo(
                f"Early stopping triggered after {patience} epochs without improvement"
            )
            break

    if best_epoch is None:
        best_epoch = history[-1]["epoch"] if history else None
        best_metrics = history[-1]["train"] if history else None

    test_metrics = None
    if test_loader is not None:
        checkpoint = _load_checkpoint(best_path, torch_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = _run_epoch(
            model,
            test_loader,
            criterion,
            torch_device,
            num_classes,
            optimizer=None,
            scaler=None,
            amp_enabled=False,
            clip_grad_norm=None,
            desc="Testing",
        )
        metrics_rows.append(_flatten_metrics_row(best_epoch or epochs, "test", test_metrics, current_lr))
        if writer is not None:
            writer.add_scalar("loss/test", test_metrics["loss"], best_epoch or epochs)
            writer.add_scalar("dice/test", test_metrics["macro"]["dice"], best_epoch or epochs)

    if writer is not None:
        writer.close()

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    _save_metadata(
        config,
        num_classes,
        history,
        best_path,
        best_epoch,
        best_metrics,
        test_metrics,
    )
    click.echo(f"✅ Training complete. Best model saved to {best_path}")
