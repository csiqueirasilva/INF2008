from __future__ import annotations

import tempfile
import csv
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
import torch

from .root import cli
from .rebuild_overlay import generate_overlay
from .train_unet import UNet, _load_checkpoint


def _resolve_device(requested: Optional[str]) -> torch.device:
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(f"Invalid device '{requested}': {exc}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("CUDA was requested but is not available on this machine")
    return device


def _load_image(path: Path) -> np.ndarray:
    array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise click.ClickException(f"Unable to read image at {path}")
    return array


def _save_mask(mask: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), mask):
        raise click.ClickException(f"Failed to write mask PNG to {destination}")


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def _load_model(model_path: Path, device: torch.device) -> tuple[UNet, int]:
    checkpoint = _load_checkpoint(model_path, device)
    state_dict = checkpoint.get("model_state_dict")
    num_classes = checkpoint.get("num_classes")
    if state_dict is None or num_classes is None:
        raise click.ClickException("Checkpoint missing required keys ('model_state_dict', 'num_classes')")

    model = UNet(n_channels=1, n_classes=int(num_classes)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, int(num_classes)


def _predict_image(
    model: UNet,
    device: torch.device,
    image_path: Path,
    overlay_out: Path,
    mask_out: Path | None,
    alpha: float,
    clahe: bool,
) -> dict:
    image = _load_image(image_path)
    processed = image
    if clahe:
        processed = _apply_clahe(image)
        click.echo("✨ Applied CLAHE preprocessing")

    tensor = torch.from_numpy(processed.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        prediction = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

    if prediction.shape != processed.shape:
        raise click.ClickException("Predicted mask shape does not match input image dimensions")

    unique_ids, counts = np.unique(prediction, return_counts=True)
    total_pixels = prediction.size
    fg_entries = []
    background_pixels = 0
    for lid, count in zip(unique_ids.tolist(), counts.tolist()):
        if lid == 0:
            background_pixels = count
            continue
        percentage = (count / total_pixels) * 100.0 if total_pixels else 0.0
        fg_entries.append({
            "label": lid,
            "pixels": count,
            "percentage": percentage,
        })

    if mask_out is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="spine_unet_"))
        mask_path = tmp_dir / "mask.png"
        cleanup_mask = True
    else:
        mask_path = mask_out
        cleanup_mask = False

    overlay_out.parent.mkdir(parents=True, exist_ok=True)
    if mask_out is not None:
        mask_out.parent.mkdir(parents=True, exist_ok=True)

    clahe_image_path: Path | None = None

    try:
        _save_mask(prediction, mask_path)
        if mask_out is not None:
            click.echo(f"🖼️ Saved predicted mask to {mask_path}")

        overlay, color_mask = generate_overlay(
            image=image_path,
            mask_labels=mask_path,
            mask=None,
            labels_json=None,
            alpha=alpha,
            ignore_label_map=False,
            legend=True,
        )

        if clahe:
            base_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(color_mask, alpha, base_rgb, 1.0 - alpha, 0.0)

        if not cv2.imwrite(str(overlay_out), overlay):
            raise click.ClickException(f"Failed to write overlay to {overlay_out}")
        if clahe:
            clahe_image_path = overlay_out.with_name(f"{overlay_out.stem}_clahe.png")
            if not cv2.imwrite(str(clahe_image_path), processed):
                raise click.ClickException(f"Failed to write CLAHE image to {clahe_image_path}")
    finally:
        if cleanup_mask:
            try:
                mask_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                mask_path.parent.rmdir()
            except OSError:
                pass

    detections_summary = (
        "🔎 Detected labels -> " + ", ".join(
            f"{entry['label']}: {entry['pixels']} px ({entry['percentage']:.2f}%)" for entry in fg_entries
        )
        if fg_entries
        else None
    )

    if detections_summary is not None:
        click.echo(detections_summary)
    else:
        percentage = (background_pixels / total_pixels) * 100.0 if total_pixels else 0.0
        click.echo(f"⚠️ No foreground labels detected (background {percentage:.2f}% of pixels)")

    click.echo(f"✅ Overlay written to {overlay_out}")

    return {
        "image": str(image_path),
        "overlay": str(overlay_out),
        "mask": str(mask_out) if mask_out is not None else None,
        "detections": fg_entries,
        "background_pct": (background_pixels / total_pixels) * 100.0 if total_pixels else 0.0,
        "clahe_image": str(clahe_image_path) if clahe else None,
    }


@cli.command("predict-unet")
@click.option("--model", "model_path", type=click.Path(path_type=Path, dir_okay=False), required=True,
              help="Path to a UNet checkpoint produced by train-unet (e.g. unet_best.pt)")
@click.option("--image", "image_path", type=click.Path(path_type=Path, dir_okay=False), required=True,
              help="Grayscale image to segment")
@click.option("--overlay-out", type=click.Path(path_type=Path, dir_okay=False), required=True,
              help="Destination PNG for the blended overlay")
@click.option("--mask-out", type=click.Path(path_type=Path, dir_okay=False), default=None,
              help="Optional path to save the raw predicted mask (uint8 label ids)")
@click.option("--device", default=None, show_default=True,
              help="Device to run inference on (defaults to cuda if available)")
@click.option("--alpha", type=float, default=0.45, show_default=True,
              help="Blend factor passed to rebuild-overlay (colour mask vs grayscale)")
@click.option("--clahe/--no-clahe", default=False, show_default=True,
              help="Apply CLAHE contrast enhancement before inference")
def predict_unet(model_path: Path, image_path: Path, overlay_out: Path, mask_out: Path | None,
                 device: str | None, alpha: float, clahe: bool) -> None:
    """Run the trained UNet on a single image and produce an overlay preview."""

    model_path = model_path.expanduser().resolve()
    image_path = image_path.expanduser().resolve()
    overlay_out = overlay_out.expanduser().resolve()
    mask_out = mask_out.expanduser().resolve() if mask_out else None

    torch_device = _resolve_device(device)

    model, _ = _load_model(model_path, torch_device)

    _predict_image(
        model=model,
        device=torch_device,
        image_path=image_path,
        overlay_out=overlay_out,
        mask_out=mask_out,
        alpha=alpha,
        clahe=clahe,
    )


@cli.command("predict-unet-batch")
@click.option("--model", "model_path", type=click.Path(path_type=Path, dir_okay=False), required=True,
              help="Path to a UNet checkpoint produced by train-unet")
@click.option("--image-dir", type=click.Path(path_type=Path, file_okay=False), required=True,
              help="Directory containing images to process")
@click.option("--pattern", default="*.png", show_default=True,
              help="Glob pattern to select images within --image-dir")
@click.option("--overlay-dir", type=click.Path(path_type=Path, file_okay=False), required=True,
              help="Directory where overlays will be written")
@click.option("--mask-dir", type=click.Path(path_type=Path, file_okay=False), default=None,
              help="Optional directory to store predicted masks (one per image)")
@click.option("--summary-csv", type=click.Path(path_type=Path, dir_okay=False), default=None,
              help="Optional CSV summary of detections")
@click.option("--device", default=None, show_default=True,
              help="Device to run inference on (defaults to cuda if available)")
@click.option("--alpha", type=float, default=0.45, show_default=True,
              help="Blend factor passed to rebuild-overlay")
@click.option("--clahe/--no-clahe", default=False, show_default=True,
              help="Apply CLAHE contrast enhancement before inference")
def predict_unet_batch(model_path: Path, image_dir: Path, pattern: str, overlay_dir: Path, mask_dir: Path | None,
                       summary_csv: Path | None, device: str | None, alpha: float, clahe: bool) -> None:
    """Run UNet inference across all images in a directory."""

    model_path = model_path.expanduser().resolve()
    image_dir = image_dir.expanduser().resolve()
    overlay_dir = overlay_dir.expanduser().resolve()
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = mask_dir.expanduser().resolve() if mask_dir else None
    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_csv.expanduser().resolve() if summary_csv else None

    if not image_dir.exists() or not image_dir.is_dir():
        raise click.ClickException(f"Image directory not found: {image_dir}")

    images = sorted(image_dir.glob(pattern))
    images = [p for p in images if p.is_file()]
    if not images:
        raise click.ClickException(f"No files matched pattern '{pattern}' under {image_dir}")

    torch_device = _resolve_device(device)
    model, _ = _load_model(model_path, torch_device)

    results = []
    detected = 0

    click.echo(f"📂 Processing {len(images)} images from {image_dir} (CLAHE={'on' if clahe else 'off'})")

    for image_path in images:
        click.echo(f"\n📸 {image_path.name}")
        overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"
        mask_path = mask_dir / f"{image_path.stem}_mask.png" if mask_dir is not None else None

        result = _predict_image(
            model=model,
            device=torch_device,
            image_path=image_path,
            overlay_out=overlay_path,
            mask_out=mask_path,
            alpha=alpha,
            clahe=clahe,
        )
        results.append(result)
        if result["detections"]:
            detected += 1

    click.echo(f"\n📊 Completed {len(results)} images; {detected} reported foreground detections")

    if summary_csv is not None:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "overlay", "mask", "clahe_image", "detected_labels", "background_pct"])
            for result in results:
                labels_summary = ";".join(
                    f"{entry['label']}:{entry['pixels']}:{entry['percentage']:.2f}" for entry in result["detections"]
                )
                writer.writerow([
                    result["image"],
                    result["overlay"],
                    result["mask"] or "",
                    result["clahe_image"] or "",
                    labels_summary,
                    f"{result['background_pct']:.2f}",
                ])
        click.echo(f"📝 Summary CSV written to {summary_csv}")
