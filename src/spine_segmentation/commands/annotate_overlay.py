from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import click
import cv2
import numpy as np

from .root import cli
from ..core.label_colors import label_to_color


@cli.command("annotate-overlay")
@click.argument("sample_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--out-path", type=click.Path(path_type=Path), default=None,
              help="Destination image path. Defaults to <sample_dir>/overlay_annotated.png")
@click.option("--thickness", type=int, default=1, show_default=True,
              help="Bounding-box line thickness (pixels)")
@click.option("--font-scale", type=float, default=0.6, show_default=True,
              help="OpenCV font scale for label text")
@click.option("--circle-radius", type=int, default=3, show_default=True,
              help="Radius (pixels) for centroid marker; 0 disables it")
@click.option("--label-prefix", default="C", show_default=True,
              help="Prefix for label text (e.g. 'C' -> C1, C2, ...)")
def annotate_overlay(sample_dir: Path, out_path: Path | None, thickness: int, font_scale: float,
                     circle_radius: int, label_prefix: str) -> None:
    """Draw bounding boxes and label ids on top of a pseudo-lateral sample.

    SAMPLE_DIR should contain the trio {image.png, mask.png, labels.json} produced by
    ``spine build-pseudo-lateral``. The command writes a new overlay image showing the
    bounding boxes, label centroids, and label ids with the same colour scheme used during
    dataset generation.
    """

    sample_dir = sample_dir.resolve()
    if not sample_dir.exists():
        raise click.ClickException(f"Sample directory not found: {sample_dir}")

    img_path = sample_dir / "image.png"
    labels_path = sample_dir / "labels.json"

    if not img_path.exists() or not labels_path.exists():
        raise click.ClickException("Expected image.png and labels.json in the sample directory")

    if out_path is None:
        out_path = sample_dir / "overlay_annotated.png"
    else:
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    base_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if base_gray is None:
        raise click.ClickException(f"Unable to read image: {img_path}")
    canvas = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)

    with open(labels_path, "r") as fh:
        meta = json.load(fh)

    labels: Dict[str, dict] = meta.get("labels", {})
    if not labels:
        raise click.ClickException("labels.json does not contain any label metadata")

    font = cv2.FONT_HERSHEY_SIMPLEX

    present_lids: set[int] = set()

    thickness = max(1, int(thickness))

    for lid_str, info in labels.items():
        try:
            lid = int(lid_str)
        except ValueError:
            continue

        if not info.get("present"):
            continue

        bbox = info.get("bbox_2d_index")
        if not bbox:
            continue

        y0, x0, y1, x1 = [int(round(v)) for v in bbox]
        h, w = canvas.shape[:2]
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h - 1, y1))
        x0 = max(0, min(w - 1, x0))
        x1 = max(0, min(w - 1, x1))
        if y1 <= y0 or x1 <= x0:
            continue

        color = label_to_color(lid)
        present_lids.add(lid)

        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness)

        centroid = info.get("centroid_2d_index")
        if circle_radius > 0 and centroid:
            cy, cx = [int(round(v)) for v in centroid]
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(canvas, (cx, cy), circle_radius, color, thickness=-1, lineType=cv2.LINE_AA)

    if present_lids:
        padding = 8
        margin = 12
        row_gap = 6
        swatch = 14
        legend_entries = []
        legend_width = 0
        line_height = 0
        text_thickness = 1

        for lid in sorted(present_lids):
            label_text = f"{label_prefix}{lid}" if label_prefix else str(lid)
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
            entry_height = max(swatch, text_h)
            legend_width = max(legend_width, swatch + 6 + text_w)
            line_height = max(line_height, entry_height)
            legend_entries.append((lid, label_text, text_w, text_h))

        legend_height = padding * 2 + len(legend_entries) * line_height + row_gap * (len(legend_entries) - 1)
        legend_width = padding * 2 + legend_width

        legend_x = max(margin, w - legend_width - margin)
        legend_y = max(margin, margin)
        x0 = int(legend_x)
        y0 = int(legend_y)
        x1 = int(min(w, legend_x + legend_width))
        y1 = int(min(h, legend_y + legend_height))
        if x1 > x0 and y1 > y0:
            roi = canvas[y0:y1, x0:x1]
            overlay = np.zeros_like(roi)
            cv2.addWeighted(overlay, 0.65, roi, 0.35, 0, roi)

            cursor_y = y0 + padding
            for lid, label_text, text_w, text_h in legend_entries:
                color = label_to_color(lid)
                row_y = cursor_y
                row_x = x0 + padding
                swatch_y0 = row_y + (line_height - swatch) // 2
                swatch_x0 = row_x
                swatch_y1 = swatch_y0 + swatch
                swatch_x1 = swatch_x0 + swatch
                cv2.rectangle(canvas, (swatch_x0, swatch_y0), (swatch_x1, swatch_y1), color, thickness=-1)

                text_x = swatch_x1 + 6
                text_y = row_y + (line_height + text_h) // 2
                cv2.putText(canvas, label_text, (text_x, text_y), font, font_scale,
                            (255, 255, 255), thickness=text_thickness, lineType=cv2.LINE_AA)

                cursor_y += line_height + row_gap

    if not cv2.imwrite(str(out_path), canvas):
        raise click.ClickException(f"Failed to write annotated overlay to {out_path}")

    click.echo(f"✅ Wrote annotated overlay to {out_path}")
