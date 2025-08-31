# src/spine_segmentation/commands/build_bank.py
import json
import click
import torch
import numpy as np
from pathlib import Path
from .root import cli
from ..core.io import find_pairs
from ..core.descriptors import hog_desc
from ..core.drr_utils import make_subject_from_paths, render_drr, device, drr_edges
from ..core.overlay import save_u8_gray

def _sample_views(mode, n, rng):
    if mode == "ap":
        yaws   = rng.uniform(-15, 15,  n)
        pitchs = rng.uniform(-20, 20,  n)
        rolls  = rng.uniform( -8,  8,  n)
    elif mode == "lateral":
        signs  = rng.choice([-1.0, 1.0], size=n)
        yaws   = signs * rng.uniform(80, 100, n)   # tighter around 90°
        pitchs = rng.uniform(-10, 10,  n)
        rolls  = rng.uniform( -6,  6,  n)
    else:  # wide
        yaws   = rng.uniform(-60, 60, n)
        pitchs = rng.uniform(-60, 60, n)
        rolls  = rng.uniform(-20, 20, n)
    return yaws, pitchs, rolls

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
            keep[int(a):int(b) + 1] = True
        else:
            keep[int(token)] = True
    return keep

@cli.command("build-bank")
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/CTSpine1K"), show_default=True)
@click.option("--subset", default="HNSCC-3DCT-RT", show_default=True)
@click.option("--limit-cases", type=int, default=8, show_default=True)
@click.option("--views-per-case", type=int, default=24, show_default=True)
@click.option("--height", type=int, default=512, show_default=True)
@click.option("--sdd", type=float, default=1600.0, show_default=True)
@click.option("--delx", type=float, default=1.2, show_default=True, help="Detector pixel size (mm/px)")
@click.option("--ty", type=float, default=None, help="Camera Y (mm). Default: 0.9*sdd")
@click.option("--auto-fov/--no-auto-fov", default=True, show_default=True,
              help="Pick pixel size automatically so the whole bbox fits.")
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/banks/hnscc_bank.npz"), show_default=True)
@click.option("--masked/--no-masked", default=True, show_default=True,
              help="Mask CT by segmentation before rendering DRRs (vertebrae only).")
@click.option("--labels", default="", help="Whitelist label IDs when masking, e.g. '1-7'")
@click.option("--min-voxels", type=int, default=1000, show_default=True,
              help="Skip cases where masked label voxels < this.")
@click.option("--debug-dir", type=click.Path(path_type=Path), default=None,
              help="If set, write per-view previews here (DRR + edges).")
@click.option("--unpack-dir", type=click.Path(path_type=Path), default=None,
              help="If set, also save an unpacked bank folder with descs.npy, metas.jsonl, previews/")
@click.option("--view-mode", type=click.Choice(["ap","lateral","wide"]), default="lateral", show_default=True)
def build_bank(data_root, subset, limit_cases, views_per_case, height, sdd, delx, ty, auto_fov,
               out, masked, labels, min_voxels, debug_dir, unpack_dir, view_mode):
    dev = device()
    pairs = find_pairs(data_root, subset, limit=limit_cases)

    dbg = Path(debug_dir) if debug_dir else None
    if dbg: dbg.mkdir(parents=True, exist_ok=True)
    upk = Path(unpack_dir) if unpack_dir else None
    if upk: (upk / "previews").mkdir(parents=True, exist_ok=True)

    descs, metas = [], []
    index = 0
    import cv2

    for idx_case, (vpath, lpath) in enumerate(pairs[:limit_cases], 1):
        click.echo(f"[{idx_case}/{min(limit_cases, len(pairs))}] {vpath.name}")
        subj = make_subject_from_paths(vpath, lpath, mask_volume=masked)

        # Optional label whitelist and sanity check
        if masked and labels:
            lab = subj["labelmap"].data                   # torch tensor, shape (1, X, Y, Z)
            max_id = int(lab.max().item())
            keep_np = _parse_label_spec(labels, max_id)   # numpy bool array or None
            if keep_np is not None:
                keep_t = torch.from_numpy(keep_np).to(device=lab.device)  # bool tensor [0..max_id]
                mask = keep_t[lab.long()]                 # bool mask same shape as lab
                vox = int(mask.sum().item())
                if vox < min_voxels:
                    click.echo("   ⚠️  skipped: masked region too small")
                    continue
                # apply mask to volume
                vol = subj["volume"].data
                subj["volume"].data = vol * mask.to(dtype=vol.dtype)

        rng = np.random.default_rng(1234 + idx_case)
        yaws, pitchs, rolls = _sample_views(view_mode, views_per_case, rng)
        txs = np.zeros(views_per_case, dtype=np.float32)
        tzs = np.zeros(views_per_case, dtype=np.float32)

        for k in range(views_per_case):
            img = render_drr(
                subj,
                yaw=float(yaws[k]), pitch=float(pitchs[k]), roll=float(rolls[k]),
                tx=float(txs[k]), ty=ty, tz=float(tzs[k]),
                sdd=sdd, height=height, delx=delx, dev=dev,
                auto_fov=auto_fov
            )

            d = hog_desc(img)
            descs.append(d)
            meta = dict(
                ct=str(vpath), seg=str(lpath),
                yaw=float(yaws[k]), pitch=float(pitchs[k]), roll=float(rolls[k]),
                tx=float(txs[k]), ty=float(0.9*sdd if ty is None else ty), tz=float(tzs[k]),
                sdd=float(sdd), height=int(height), delx=float(delx) if not auto_fov else float("nan"),
                auto_fov=bool(auto_fov),
                subset=subset, case=vpath.stem, view=int(k), bank_index=int(index),
                masked=bool(masked),
            )
            metas.append(meta)

            if dbg:
                case_dir = dbg / subset / vpath.stem
                case_dir.mkdir(parents=True, exist_ok=True)
                save_u8_gray(case_dir / f"{k:03d}_drr.png", img)
                save_u8_gray(case_dir / f"{k:03d}_edges.png", drr_edges(img))
                with open(case_dir / f"{k:03d}_meta.json", "w") as f:
                    json.dump(meta, f, indent=2)

            if upk:
                prev_dir = upk / "previews" / f"{index:06d}"
                prev_dir.mkdir(parents=True, exist_ok=True)
                save_u8_gray(prev_dir / "drr.png", img)
                save_u8_gray(prev_dir / "edges.png", drr_edges(img))
                with open(prev_dir / "meta.json", "w") as f:
                    json.dump(meta, f, indent=2)

            index += 1

    if not metas:
        raise click.ClickException("No views generated. Check labels/min-voxels or try --no-masked / --auto-fov.")

    descs = np.vstack(descs).astype(np.float32)
    metas = np.array(metas, dtype=object)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, descs=descs, metas=metas)
    click.echo(f"✅ Saved bank: {out} (N={len(metas)})")

    if upk:
        np.save(upk / "descs.npy", descs)
        with open(upk / "metas.jsonl", "w") as f:
            for m in metas:
                j = m.item() if isinstance(m, np.ndarray) else m
                f.write(json.dumps(j) + "\n")
        click.echo(f"📂 Unpacked bank folder: {upk}")
