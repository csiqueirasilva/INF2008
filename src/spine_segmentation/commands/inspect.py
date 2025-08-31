# src/spine_segmentation/commands/inspect.py
import json
import click
import numpy as np
from pathlib import Path
import cv2

from .root import cli
from ..core.io import load_nii
from ..core.overlay import save_u8_gray

def colorize(mask, color=(0,255,0)):
    rgb = np.zeros((*mask.shape,3), np.uint8)
    rgb[mask>0] = color
    return rgb

def _parse_label_spec(spec: str, max_id: int):
    if not spec or not spec.strip():
        return None
    keep = np.zeros(max_id + 1, dtype=bool)
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-")
            keep[int(a):int(b)+1] = True
        else:
            keep[int(token)] = True
    return keep

@cli.command("inspect")
@click.option("--ct",  type=click.Path(path_type=Path), required=True)
@click.option("--seg", type=click.Path(path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("outputs/inspect"), show_default=True)
@click.option("--labels", default="", help="Whitelist label IDs (e.g. '1-7,20-24'). Empty = keep all IDs.")
@click.option("--binarize/--no-binarize", default=False, show_default=True,
              help="If true, collapse labels>0 into a single binary mask.")
def inspect(ct, seg, out_dir, labels, binarize):
    out_dir.mkdir(parents=True, exist_ok=True)
    vol, sp, _ = load_nii(ct)     # (Z,Y,X) float HU
    lab, _, _ = load_nii(seg)     # (Z,Y,X) int
    lab = lab.astype(np.int32)

    # label whitelist (optional)
    keep = _parse_label_spec(labels, int(lab.max()))
    if keep is not None:
        lab = lab * keep[lab]

    # histogram of the (possibly whitelisted) labels
    uniq, cnt = np.unique(lab, return_counts=True)
    with open(out_dir/"label_hist.json","w") as f:
        json.dump({"spacing": sp.tolist(),
                   "hist": {int(u): int(c) for u,c in zip(uniq, cnt)}}, f, indent=2)

    # visualize with current (potentially multi-ID) mask
    zc, yc, xc = [s//2 for s in lab.shape]
    v = np.clip((vol+1000)/2000*255, 0, 255).astype(np.uint8)

    planes = [
        ("ax",  v[zc],      (lab[zc]      > 0).astype(np.uint8)*255),
        ("cor", v[:,yc,:],  (lab[:,yc,:]  > 0).astype(np.uint8)*255),
        ("sag", v[:,:,xc],  (lab[:,:,xc]  > 0).astype(np.uint8)*255),
    ]
    for name, img, mask in planes:
        save_u8_gray(out_dir/f"00_{name}_ct.png", img)
        save_u8_gray(out_dir/f"01_{name}_seg.png", mask)
        over = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.7,
                               colorize(mask), 0.3, 0.0)
        cv2.imwrite(str(out_dir/f"02_{name}_overlay.png"), over)

    # simple MIPs of masked CT (union of labels>0)
    union = (lab > 0).astype(np.uint8)
    save_u8_gray(out_dir/"10_mip_ax.png",  (v * union).max(axis=0))
    save_u8_gray(out_dir/"11_mip_cor.png", (v * union).max(axis=1))
    save_u8_gray(out_dir/"12_mip_sag.png", (v * union).max(axis=2))

    # connected components
    from skimage.measure import label as cc_label, regionprops

    # union components (treat all labels>0 as one mask)
    cc_u = cc_label(union, connectivity=1)
    props_u = regionprops(cc_u)
    comps_u = []
    for p in props_u:
        zci, yci, xci = p.centroid
        comps_u.append({
            "label": int(p.label),
            "voxels": int(p.area),
            "centroid_index": [zci, yci, xci],
            "centroid_mm": [zci*sp[2], yci*sp[1], xci*sp[0]],
            "bbox_index": [int(v) for v in p.bbox],
        })
    with open(out_dir/"components_union.json","w") as f:
        json.dump({"shape": list(lab.shape), "spacing": sp.tolist(), "components": comps_u}, f, indent=2)

    # per-label components (keeps IDs 1..N separate)
    comp_by_label = {}
    for lid in sorted(int(u) for u in uniq if u != 0):
        m = (lab == lid).astype(np.uint8)
        cc = cc_label(m, connectivity=1)
        props = regionprops(cc)
        comp_by_label[str(lid)] = [{
            "voxels": int(p.area),
            "centroid_index": list(p.centroid),
            "centroid_mm": [p.centroid[0]*sp[2], p.centroid[1]*sp[1], p.centroid[2]*sp[0]],
            "bbox_index": [int(v) for v in p.bbox],
        } for p in props]
    with open(out_dir/"components_by_label.json","w") as f:
        json.dump({"shape": list(lab.shape), "spacing": sp.tolist(),
                   "labels": comp_by_label}, f, indent=2)

    # optional binarize view for quick human checks
    if binarize:
        save_u8_gray(out_dir/"99_binarized_union.png", union.astype(np.uint8)*255)

    click.echo(f"📂 Inspect outputs in {out_dir}")
