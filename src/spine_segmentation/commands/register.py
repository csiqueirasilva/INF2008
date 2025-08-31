import json, numpy as np
import click
from pathlib import Path

from ..core.preproc import preprocess_frame
from ..core.drr_utils import make_subject_from_paths, make_drr, project_binary_mask, device
from ..core.registration import refine_pose
from ..core.overlay import overlay_mask_on_gray, save_u8_gray
from .root import cli

import torch

@cli.command("register")
@click.option("--bank", type=click.Path(path_type=Path), default=Path("data/banks/hnscc_bank.npz"), show_default=True)
@click.option("--frame", type=click.Path(path_type=Path), required=True, help="Path to your 2D lateral image")
@click.option("--out", type=click.Path(path_type=Path), default=Path("outputs/overlay.png"), show_default=True)
@click.option("--iters", type=int, default=250, show_default=True)
@click.option("--topk", type=int, default=5, show_default=True)
@click.option("--debug-dir", type=click.Path(path_type=Path), default=None,
              help="If set, save coarse/refined DRRs and blends here")
def register(bank, frame, out, iters, topk, debug_dir):
    dev = device()
    out.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(bank, allow_pickle=True)
    descs = data["descs"]
    metas = list(data["metas"])

    from ..core.descriptors import hog_desc
    qry_img = preprocess_frame(frame, size=384)

    import cv2
    dbg = Path(debug_dir) if debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)
        # grayscale-safe write
        save_u8_gray(dbg / "00_query.png", qry_img)

    q = hog_desc(qry_img)

    sims = np.dot(descs, q) / (np.linalg.norm(descs, axis=1)*np.linalg.norm(q) + 1e-8)
    top_idx = np.argsort(-sims)[:topk]

    best_pack = None
    best_loss = float("inf")
    qry_t = torch.from_numpy((qry_img/255.0).astype(np.float32))[None,None,...].to(dev)

    restarts = 3

    from ..core.drr_utils import render_drr, mask_to_outline

    for i in top_idx:
        m = metas[i].item() if isinstance(metas[i], np.ndarray) else metas[i]
        subj = make_subject_from_paths(Path(m["ct"]))

        auto_fov = bool(m.get("auto_fov", False))
        delx = m.get("delx", float("nan"))
        try:
            import math
            finite_delx = math.isfinite(float(delx))
        except Exception:
            finite_delx = False
        if dbg:
            coarse = render_drr(
                subj,
                yaw=float(m["yaw"]), pitch=float(m["pitch"]), roll=float(m["roll"]),
                tx=float(m["tx"]), ty=float(m["ty"]), tz=float(m["tz"]),
                sdd=m["sdd"], height=m["height"], delx=(float(delx) if finite_delx else 1.0), dev=dev,
                auto_fov=auto_fov or not finite_delx,
            )
            # grayscale-safe write
            save_u8_gray(dbg / f"{len(list(dbg.glob('*.png'))):02d}_coarse_sim{float(sims[i]):.3f}.png", coarse)

        # Build DRR for refinement; if bank used auto_fov, recompute delx.
        if auto_fov or not finite_delx:
            # lazy import to avoid circulars; compute auto delx same way as bank build
            from ..core.drr_utils import _auto_delx_from_subject as _auto_delx
            delx_ref = _auto_delx(subj, height=m["height"])  # consistent with current subject
        else:
            delx_ref = float(delx)
        drr = make_drr(subj, sdd=m["sdd"], height=m["height"], delx=delx_ref, dev=dev)

        rot0 = torch.tensor([[m["yaw"], m["pitch"], m["roll"]]], device=dev, dtype=torch.float32)
        trs0 = torch.tensor([[m["tx"],  m["ty"],   m["tz"]  ]], device=dev, dtype=torch.float32)

        for _ in range(restarts):
            rot_seed = rot0 + torch.randn_like(rot0)*0.03   # ~2°
            trs_seed = trs0 + torch.randn_like(trs0)*15.0   # mm
            rot_f, trs_f, loss = refine_pose(drr, qry_t, rot_seed, trs_seed, iters=iters, lr=0.02, dev=dev)
            if dbg:
                refined = render_drr(
                    subj,
                    yaw=float(rot_f[0,0].item()), pitch=float(rot_f[0,1].item()), roll=float(rot_f[0,2].item()),
                    tx=float(trs_f[0,0].item()), ty=float(trs_f[0,1].item()), tz=float(trs_f[0,2].item()),
                    sdd=m["sdd"], height=m["height"], delx=(float(delx) if finite_delx else 1.0), dev=dev,
                    auto_fov=auto_fov or not finite_delx,
                )
                # refined
                save_u8_gray(dbg / f"{len(list(dbg.glob('*.png'))):02d}_refined_loss{loss:.4f}.png", refined)

        if loss < best_loss:
            best_loss = loss
            best_pack = (subj, Path(m["seg"]), rot_f.detach().clone(), trs_f.detach().clone(), m, delx_ref)

    if best_pack is None:
        raise click.ClickException("No candidate survived refinement.")

    subj, seg_path, rot, trs, m, best_delx = best_pack

    # save best refined DRR
    # Re-render best with the same auto_fov handling
    auto_fov = bool(m.get("auto_fov", False))
    delx = m.get("delx", float("nan"))
    try:
        import math
        finite_delx = math.isfinite(float(delx))
    except Exception:
        finite_delx = False
    best_refined = render_drr(
        subj,
        yaw=float(rot[0,0].item()), pitch=float(rot[0,1].item()), roll=float(rot[0,2].item()),
        tx=float(trs[0,0].item()), ty=float(trs[0,1].item()), tz=float(trs[0,2].item()),
        sdd=m["sdd"], height=m["height"], delx=(float(delx) if finite_delx else 1.0), dev=dev,
        auto_fov=auto_fov or not finite_delx,
    )

    if dbg:
        save_u8_gray(dbg / "99_best_refined_drr.png", best_refined)
        ref_for_blend = best_refined
        if ref_for_blend.shape != qry_img.shape:
            import cv2
            ref_for_blend = cv2.resize(ref_for_blend, (qry_img.shape[1], qry_img.shape[0]), interpolation=cv2.INTER_AREA)
            save_u8_gray(dbg / "98_best_refined_resized.png", ref_for_blend)
        blend = cv2.addWeighted(cv2.cvtColor(qry_img, cv2.COLOR_GRAY2BGR), 0.6,
                                cv2.cvtColor(ref_for_blend, cv2.COLOR_GRAY2BGR), 0.4, 0.0)
        cv2.imwrite(str(dbg / "99_best_blend.png"), blend)

    mask = project_binary_mask(
        subject_ct=subj,
        seg_path=seg_path,
        rot=rot, trs=trs,
        sdd=m["sdd"], height=m["height"], delx=float(best_delx), dev=dev
    )
    outline = mask_to_outline(mask, k=3)
    overlay = overlay_mask_on_gray(qry_img, outline, alpha=0.9, color=(0,255,0))
    out = Path(out)
    cv2.imwrite(str(out), overlay)

    # dump masks if debug dir is provided (helps manual comparison)
    if dbg:
        save_u8_gray(dbg / "95_projected_mask.png", mask)
        save_u8_gray(dbg / "96_projected_outline.png", outline)

    pose_path = out.with_suffix(".pose.json")
    with open(pose_path, "w") as f:
        json.dump({
            "rot": rot.cpu().numpy().tolist(),
            "trs": trs.cpu().numpy().tolist(),
            "meta": m,
            "loss": float(best_loss)
        }, f, indent=2)

    click.echo(f"✅ Saved overlay: {out}")
    click.echo(f"   Pose & meta:  {pose_path}")
