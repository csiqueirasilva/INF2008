from __future__ import annotations

from pathlib import Path
from typing import Dict

import click
import cv2
import numpy as np

from .root import cli
from ..core.label_colors import label_to_color


@cli.command("colorize-labels")
@click.argument("sample_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--out-path", type=click.Path(path_type=Path), default=None,
              help="Destination path for the blended overlay (defaults to overlay_recolored.png)")
@click.option("--mask-out", type=click.Path(path_type=Path), default=None,
              help="Optional path to save the colourised label mask only")
@click.option("--alpha", type=float, default=0.45, show_default=True,
              help="Blend factor for the colour mask over the grayscale image")
@click.option("--label-prefix", default="C", show_default=True,
              help="Prefix used when reporting labels in logs")
def colorize_labels(sample_dir: Path, out_path: Path | None, mask_out: Path | None,
                    alpha: float, label_prefix: str) -> None:
    """Rebuild a coloured overlay using the stored per-label mask and image."""

    sample_dir = sample_dir.resolve()
    img_path = sample_dir / "image.png"
    label_map_path = sample_dir / "mask_labels.png"

    if not img_path.exists():
        raise click.ClickException(f"image.png not found under {sample_dir}")
    if not label_map_path.exists():
        raise click.ClickException(
            f"mask_labels.png missing under {sample_dir}. Re-run pseudo-lateral generation with the updated pipeline."
        )

    image_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise click.ClickException(f"Could not read image {img_path}")

    labels = cv2.imread(str(label_map_path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise click.ClickException(f"Could not read label map {label_map_path}")
    if labels.shape[:2] != image_gray.shape[:2]:
        raise click.ClickException("mask_labels.png shape does not match image.png")

    h, w = labels.shape[:2]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = sorted(int(v) for v in np.unique(labels) if v > 0)
    if not unique_labels:
        click.echo("⚠️ No label ids found in mask_labels.png")

    for lid in unique_labels:
        color = label_to_color(lid)
        color_mask[labels == lid] = color

    if mask_out is not None:
        mask_out = Path(mask_out).resolve()
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_out), color_mask):
            raise click.ClickException(f"Failed to write colour mask to {mask_out}")

    if out_path is None:
        out_path = sample_dir / "overlay_recolored.png"
    else:
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(color_mask, alpha, image_color, 1.0 - alpha, 0.0)

    if not cv2.imwrite(str(out_path), blended):
        raise click.ClickException(f"Failed to write blended overlay to {out_path}")

    click.echo(f"✅ Wrote colour overlay to {out_path}")
    if mask_out is not None:
        click.echo(f"   🎨 Colour mask saved to {mask_out}")
