import json
import click
import numpy as np
from pathlib import Path

from .root import cli
from ..core.drr_utils import make_subject_from_paths, render_drr, drr_edges, device
from ..core.overlay import save_u8_gray

@cli.command("bank-export")
@click.option("--bank", type=click.Path(path_type=Path), required=True, help=".npz bank")
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("outputs/bank_export"), show_default=True)
@click.option("--limit", type=int, default=0, show_default=True, help="Max items to export (0 = all)")
def bank_export(bank, out_dir, limit):
    """Re-render masked DRR previews for each bank item; write unpacked folder."""
    dev = device()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(bank, allow_pickle=True)
    metas = list(data["metas"])
    if limit > 0:
        metas = metas[:limit]

    for i, m in enumerate(metas):
        m = m.item() if isinstance(m, np.ndarray) else m
        subj = make_subject_from_paths(m["ct"], m["seg"], mask_volume=m.get("masked", True))
        # Some banks were built with auto FOV, storing delx as NaN. Recompute consistently.
        auto_fov = bool(m.get("auto_fov", False))
        delx = m.get("delx", float("nan"))
        try:
            import math
            finite_delx = math.isfinite(float(delx))
        except Exception:
            finite_delx = False

        img = render_drr(
            subj,
            yaw=m["yaw"], pitch=m["pitch"], roll=m["roll"],
            tx=m["tx"],  ty=m["ty"],   tz=m["tz"],
            sdd=m["sdd"], height=m["height"], delx=(float(delx) if finite_delx else 1.0), dev=dev,
            auto_fov=auto_fov or not finite_delx,
        )
        prev_dir = out_dir / "previews" / f"{i:06d}"
        prev_dir.mkdir(parents=True, exist_ok=True)
        save_u8_gray(prev_dir / "drr.png", img)
        save_u8_gray(prev_dir / "edges.png", drr_edges(img))
        with open(prev_dir / "meta.json", "w") as f:
            json.dump(m, f, indent=2)

    # write metas.jsonl for convenience
    with open(out_dir / "metas.jsonl", "w") as f:
        for i, mm in enumerate(metas):
            m = mm.item() if isinstance(mm, np.ndarray) else mm
            f.write(json.dumps(m) + "\n")

    click.echo(f"📂 Exported previews to {out_dir}")
