# src/spine_segmentation/commands/frames_manifest.py
import click, os, glob
from pathlib import Path
import pandas as pd

from .root import cli

def _find_unique_masks(root_dir: Path) -> list[Path]:
    # Recursively find Mask.tif; keep the first per parent folder name
    paths = glob.glob(os.path.join(str(root_dir), "**", "Mask.tif"), recursive=True)
    uniq = {}
    for p in paths:
        p = Path(p)
        key = p.parent.name  # folder name as key
        uniq.setdefault(key, p)
    return list(uniq.values())

@cli.command("frames-manifest")
@click.option("--frames-dir", type=click.Path(path_type=Path), default=Path("data/frames"), show_default=True)
@click.option("--manifest", type=click.Path(path_type=Path), default=Path("data/frames_manifest.csv"), show_default=True)
@click.option("--masks-root", type=click.Path(path_type=Path), default=None, help="Root to search for Mask.tif")
@click.option("--write", type=click.Path(path_type=Path), default=Path("data/frames_metadata.csv"), show_default=True)
def frames_manifest(frames_dir, manifest, masks_root, write):
    """
    Build/refresh a frames DataFrame and (optionally) attach Mask.tif labels.
    - If --manifest exists, it will be used; otherwise frames_dir is scanned.
    - Mask matching uses the frame's base name (without extension) against the
      parent folder name of each Mask.tif (same strategy your teammate used).
    """
    if Path(manifest).exists():
        df = pd.read_csv(manifest)
        # ensure 'out_path' column exists (from our extractor); if not, fallback to scan
        if "out_path" in df.columns:
            paths = df["out_path"].tolist()
        else:
            paths = []
    else:
        # fallback: scan frames_dir
        paths = sorted(list(Path(frames_dir).glob("**/*.png"))) + sorted(list(Path(frames_dir).glob("**/*.jpg")))
        df = pd.DataFrame({"out_path": [str(p) for p in paths]})

    if not len(df):
        raise click.ClickException("No frames found to index.")

    df["label"] = "unlabeled"
    if masks_root:
        masks = _find_unique_masks(Path(masks_root))
        mask_dict = {Path(p).parent.name: str(p) for p in masks}
        def att(row):
            base = Path(row["out_path"]).stem  # e.g., vX_f000123
            return mask_dict.get(base, "unlabeled")
        df["label"] = df.apply(att, axis=1)

    write.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(write, index=False)
    click.echo(f"📝 Wrote {write}  (rows: {len(df)})")
    if masks_root:
        n_lab = int((df["label"] != "unlabeled").sum())
        click.echo(f"🔗 Attached masks: {n_lab}")
