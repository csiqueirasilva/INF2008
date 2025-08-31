import csv
import json
from pathlib import Path
import click
import numpy as np
import torch

from .root import cli
from ..core.io import find_pairs
from ..core.drr_utils import make_subject_from_paths, render_drr, project_binary_mask, device
from ..core.overlay import save_u8_gray


def _sample_views(mode, n, rng):
    if mode == "ap":
        yaws = rng.uniform(-15, 15, n)
        pitchs = rng.uniform(-20, 20, n)
        rolls = rng.uniform(-8, 8, n)
    elif mode == "lateral":
        signs = rng.choice([-1.0, 1.0], size=n)
        yaws = signs * rng.uniform(80, 100, n)
        pitchs = rng.uniform(-10, 10, n)
        rolls = rng.uniform(-6, 6, n)
    else:
        yaws = rng.uniform(-60, 60, n)
        pitchs = rng.uniform(-60, 60, n)
        rolls = rng.uniform(-20, 20, n)
    return yaws, pitchs, rolls


@cli.command("build-synth")
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/CTSpine1K"), show_default=True)
@click.option("--subset", default="HNSCC-3DCT-RT", show_default=True)
@click.option("--limit-cases", type=int, default=20, show_default=True)
@click.option("--views-per-case", type=int, default=30, show_default=True)
@click.option("--height", type=int, default=512, show_default=True)
@click.option("--sdd", type=float, default=1600.0, show_default=True)
@click.option("--ty", type=float, default=None, help="Camera Y (mm). Default: 0.9*sdd")
@click.option("--auto-fov/--no-auto-fov", default=True, show_default=True)
@click.option("--view-mode", type=click.Choice(["ap","lateral","wide"]), default="lateral", show_default=True)
@click.option("--masked/--no-masked", default=True, show_default=True, help="Mask CT by segmentation")
@click.option("--labels", default="", help="Whitelist label IDs when masking, e.g. '1-7'")
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("data/synth2d"), show_default=True)
@click.option("--val-ratio", type=float, default=0.1, show_default=True)
@click.option("--min-voxels", type=int, default=1000, show_default=True, help="Skip cases where masked label voxels < this.")
@click.option("--min-std", type=float, default=1.0, show_default=True, help="Skip rendered views with std-dev below this (flat/black).")
@click.option("--clean/--no-clean", default=False, show_default=True, help="Wipe out_dir before generating (avoids stale files)")
@click.option("--debug-dir", type=click.Path(path_type=Path), default=None, help="If set, write per-view previews and meta here.")
@click.option("--attempts-per-view", type=int, default=8, show_default=True, help="Resample pose up to N times until a view passes quality checks.")
def build_synth(data_root, subset, limit_cases, views_per_case, height, sdd, ty, auto_fov,
                view_mode, masked, labels, out_dir, val_ratio, min_voxels, min_std, clean, debug_dir, attempts_per_view):
    """
    Generate a synthetic 2D dataset of DRRs + projected masks for training a 2D segmenter.
    Writes images/, masks/, manifest.csv with split train/val.
    """
    dev = device()
    out_dir = Path(out_dir)
    if clean and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    dbg = Path(debug_dir) if debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)

    from ..core.commands_utils import parse_label_spec  # will add tiny util below

    pairs = find_pairs(data_root, subset, limit=0)
    sel_pairs = pairs if (limit_cases == 0) else pairs[:limit_cases]
    rng = np.random.default_rng(42)

    rows = []
    idx = 0
    n_skipped_cases = 0
    n_skipped_views_flat = 0

    for ci, (ct_path, seg_path) in enumerate(sel_pairs, 1):
        click.echo(f"[{ci}/{len(sel_pairs)}] {ct_path.name}")
        subj = make_subject_from_paths(ct_path, seg_path, mask_volume=masked)
        has_seg = (seg_path is not None) and ("labelmap" in subj)
        if not has_seg:
            click.echo("   ⚠️  no segmentation found; skipping case (no GT mask)")
            n_skipped_cases += 1
            continue
        # Optional label whitelist and sanity check
        lab = subj["labelmap"].data
        max_id = int(lab.max().item())
        if labels:
            keep_np = parse_label_spec(labels, max_id)
            if keep_np is not None:
                keep_t = torch.from_numpy(keep_np).to(device=lab.device)
                m = keep_t[lab.long()]  # bool mask
                vox = int(m.sum().item())
                if vox < min_voxels:
                    click.echo("   ⚠️  masked voxels too small; skipping case")
                    n_skipped_cases += 1
                    continue
                subj["volume"].data *= m.to(dtype=subj["volume"].data.dtype)
        else:
            # no whitelist: validate union label size if masking is requested
            if masked:
                m = (lab > 0)
                vox = int(m.sum().item())
                if vox < min_voxels:
                    click.echo("   ⚠️  union label too small; skipping case")
                    n_skipped_cases += 1
                    continue

        tx0 = 0.0; tz0 = 0.0
        for k in range(views_per_case):
            accepted = False
            for attempt in range(1, attempts_per_view + 1):
                # sample a pose
                yaw, pitch, roll = [float(v) for v in _sample_views(view_mode, 1, rng)]
                try:
                    yaw = float(yaw[0]); pitch = float(pitch[0]); roll = float(roll[0])
                except Exception:
                    pass

                # ensure consistent delx between DRR and mask projection
                if auto_fov:
                    from ..core.drr_utils import _auto_delx_from_subject as _auto_delx
                    delx_use = float(_auto_delx(subj, height))
                else:
                    delx_use = 1.2

                img = render_drr(
                    subj,
                    yaw=yaw, pitch=pitch, roll=roll,
                    tx=tx0, ty=ty, tz=tz0,
                    sdd=sdd, height=height, delx=delx_use, dev=dev, auto_fov=False
                )
                # Skip flat/black renders
                img_std = float(np.std(img))
                if dbg:
                    from ..core.overlay import save_u8_gray as _save
                    case_dbg = dbg / f"{subset}_{ct_path.stem}"
                    case_dbg.mkdir(exist_ok=True, parents=True)
                    _save(case_dbg / f"{k:04d}_a{attempt:02d}_img.png", img)
                if float(img_std) < float(min_std):
                    n_skipped_views_flat += 1
                    if dbg:
                        with open(case_dbg / f"{k:04d}_a{attempt:02d}_meta.json", "w") as f:
                            json.dump({"reason": "flat_image", "std": img_std, "delx": delx_use,
                                       "yaw": yaw, "pitch": pitch, "roll": roll}, f, indent=2)
                    continue
                # build rot/trs tensors for the seg projection
                rot = torch.tensor([[yaw, pitch, roll]], device=dev)
                trs = torch.tensor([[tx0, float(0.9*sdd if ty is None else ty), tz0]], device=dev)
                mask = project_binary_mask(subj, seg_path, rot, trs, sdd=sdd, height=height, delx=delx_use, dev=dev)
                if dbg:
                    _save(case_dbg / f"{k:04d}_a{attempt:02d}_mask.png", mask)
                if int(mask.max()) == 0:
                    # no projected thickness => useless label
                    n_skipped_views_flat += 1
                    if dbg:
                        with open(case_dbg / f"{k:04d}_a{attempt:02d}_meta.json", "w") as f:
                            json.dump({"reason": "empty_mask", "std": img_std, "delx": delx_use,
                                       "yaw": yaw, "pitch": pitch, "roll": roll}, f, indent=2)
                    continue

                # accept this attempt
                img_path = out_dir / "images" / f"{subset}_{ct_path.stem}_{k:04d}.png"
                msk_path = out_dir / "masks"  / f"{subset}_{ct_path.stem}_{k:04d}.png"
                save_u8_gray(img_path, img)
                save_u8_gray(msk_path, mask)
                if dbg:
                    with open(case_dbg / f"{k:04d}_a{attempt:02d}_meta.json", "w") as f:
                        json.dump({"reason": "accepted", "std": img_std, "delx": delx_use,
                                   "yaw": yaw, "pitch": pitch, "roll": roll}, f, indent=2)
                split = "val" if (rng.random() < val_ratio) else "train"
                rows.append(dict(image=str(img_path), mask=str(msk_path), split=split,
                                 subset=subset, case=ct_path.stem, yaw=yaw, pitch=pitch, roll=roll))
                idx += 1
                accepted = True
                break
            if not accepted:
                click.echo(f"   ⚠️  view {k} rejected after {attempts_per_view} attempts")

    if not rows:
        raise click.ClickException("No samples generated. Check labels/seg availability.")

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "data_root": str(data_root), "subset": subset, "views_per_case": views_per_case,
            "height": height, "sdd": sdd, "auto_fov": bool(auto_fov), "masked": bool(masked), "labels": labels,
            "total": len(rows), "skipped_cases": int(n_skipped_cases), "skipped_views_flat": int(n_skipped_views_flat)
        }, f, indent=2)
    click.echo(f"✅ Wrote dataset to {out_dir} (N={len(rows)})")
