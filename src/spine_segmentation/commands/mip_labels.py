from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple
import click
import numpy as np
import cv2

from .root import cli
from ..core.io import find_pairs, load_nii


def _ct_to_u8(vol: np.ndarray) -> np.ndarray:
    # Basic bone-ish window from HU without requiring external libs
    v = np.clip((vol + 1000.0) / 2000.0 * 255.0, 0, 255).astype(np.uint8)
    return v


def _mip_and_spacing(vol_u8: np.ndarray, lab: np.ndarray, spacing: np.ndarray, plane: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    # vol_u8: (Z,Y,X) uint8; lab: (Z,Y,X) int; spacing: (sx, sy, sz) from nib header
    # Return: base_mip_u8, union_mask_mip_u8, pixel_spacing for the 2D plane
    if plane == "sag":
        base = (vol_u8 * (lab > 0)).max(axis=2)  # (Z,Y)
        union = (lab > 0).max(axis=2).astype(np.uint8) * 255
        pix = (float(spacing[2]), float(spacing[1]))  # (sz, sy) -> aligns with (Z,Y)
    elif plane == "cor":
        base = (vol_u8 * (lab > 0)).max(axis=1)  # (Z,X)
        union = (lab > 0).max(axis=1).astype(np.uint8) * 255
        pix = (float(spacing[2]), float(spacing[0]))  # (sz, sx) -> aligns with (Z,X)
    elif plane == "ax":
        base = (vol_u8 * (lab > 0)).max(axis=0)  # (Y,X)
        union = (lab > 0).max(axis=0).astype(np.uint8) * 255
        pix = (float(spacing[1]), float(spacing[0]))  # (sy, sx) -> aligns with (Y,X)
    else:
        raise click.ClickException(f"Invalid plane: {plane}")
    return base, union, pix


def _label_silhouette(lab: np.ndarray, lid: int, plane: str) -> np.ndarray:
    m = (lab == int(lid)).astype(np.uint8)
    if plane == "sag":
        sil = m.max(axis=2)
    elif plane == "cor":
        sil = m.max(axis=1)
    else:  # ax
        sil = m.max(axis=0)
    return (sil > 0).astype(np.uint8) * 255


def _centroid_2d(mask_u8: np.ndarray) -> Tuple[float, float] | None:
    idx = np.argwhere(mask_u8 > 0)
    if idx.size == 0:
        return None
    z_or_y = float(idx[:, 0].mean())
    y_or_x = float(idx[:, 1].mean())
    return (z_or_y, y_or_x)


def _bbox_2d(mask_u8: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return (y0, x0, y1, x1)


def _overlay_multi(gray_u8: np.ndarray, masks: dict[int, np.ndarray], cents: dict[int, Tuple[float,float]]|None=None) -> np.ndarray:
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    palette = {
        1: (255, 0, 0),
        2: (255, 128, 0),
        3: (255, 255, 0),
        4: (0, 255, 0),
        5: (0, 255, 255),
        6: (0, 128, 255),
        7: (0, 0, 255),
    }
    overlay = np.zeros_like(base)
    for lid, m in masks.items():
        if m is None:
            continue
        color = palette.get(int(lid), (255, 255, 255))
        mk = (m > 0)
        overlay[mk] = color
    out = cv2.addWeighted(base, 1.0, overlay, 0.35, 0.0)
    if cents:
        for lid, c in cents.items():
            if c is None:
                continue
            y, x = int(round(c[0])), int(round(c[1]))
            color = palette.get(int(lid), (255, 255, 255))
            cv2.circle(out, (x, y), 3, color, -1)
            cv2.putText(out, str(lid), (x+4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


@cli.command("mip-labels")
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/CTSpine1K"), show_default=True)
@click.option("--subset", default="HNSCC-3DCT-RT", show_default=True)
@click.option("--limit-cases", type=int, default=0, show_default=True, help="0 = all")
@click.option("--labels", default="1-7", show_default=True, help="Label IDs to include (e.g., '1-7,20-24')")
@click.option("--plane", type=click.Choice(["sag", "cor", "ax"]), default="sag", show_default=True)
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("outputs/mip_labels"), show_default=True)
def mip_labels(data_root, subset, limit_cases, labels, plane, out_dir):
    """
    Generate simple MIP images with C1..C7 (or chosen labels) marked.
    Writes per-case overlay images, per-label silhouettes, and a JSON with centroids/bboxes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(data_root, subset, limit=(None if limit_cases == 0 else limit_cases))
    if not pairs:
        raise click.ClickException("No cases found.")

    processed = 0
    for i, (ct, seg) in enumerate(pairs, 1):
        if seg is None:
            click.echo(f"[{i}] {ct.name}: no seg; skipping")
            continue
        click.echo(f"[{i}] {ct.name}")
        vol, sp, _ = load_nii(ct)
        lab, _, _ = load_nii(seg)
        vol_u8 = _ct_to_u8(vol)
        lab = lab.astype(np.int32)

        base, union, pix = _mip_and_spacing(vol_u8, lab, sp, plane)

        # build requested label set
        lids = []
        for tok in str(labels).split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                a, b = tok.split("-")
                lids.extend(list(range(int(a), int(b) + 1)))
            else:
                lids.append(int(tok))
        lids = sorted(set(lids))

        # per-label silhouettes and stats
        masks2d = {}
        stats = {}
        cents2d = {}
        for lid in lids:
            sil = _label_silhouette(lab, lid, plane)
            if sil.max() == 0:
                masks2d[lid] = None
                continue
            masks2d[lid] = sil
            c2 = _centroid_2d(sil)
            bb = _bbox_2d(sil)
            cents2d[lid] = c2
            stats[str(lid)] = {
                "present": True,
                "centroid_2d_index": (None if c2 is None else [float(c2[0]), float(c2[1])]),
                "bbox_2d_index": (None if bb is None else [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]),
            }

        # overlay image
        over = _overlay_multi(base, masks2d, cents2d)

        case_dir = out_dir / f"{subset}_{ct.stem}"
        case_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(case_dir / f"mip_{plane}.png"), base)
        cv2.imwrite(str(case_dir / f"overlay_{plane}.png"), over)
        for lid in lids:
            sil = masks2d.get(lid)
            if sil is not None:
                cv2.imwrite(str(case_dir / f"label_{lid:02d}_{plane}.png"), sil)

        with open(case_dir / "labels.json", "w") as f:
            json.dump({
                "case": ct.stem,
                "subset": subset,
                "plane": plane,
                "image": str(case_dir / f"overlay_{plane}.png"),
                "pixel_spacing": list(pix),  # (row_mm, col_mm)
                "labels": stats,
            }, f, indent=2)

        processed += 1

    click.echo(f"✅ Processed {processed} cases to {out_dir}")
