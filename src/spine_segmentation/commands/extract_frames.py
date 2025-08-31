# src/spine_segmentation/commands/extract_frames.py
import click
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from .root import cli
from ..core.video import extract_frames_from_folder

@cli.command("extract-frames")
@click.option("--input-dir",  type=click.Path(path_type=Path), required=True, help="Folder with .avi videos")
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/frames"), show_default=True)
@click.option("--ext", type=click.Choice(["png","jpg"], case_sensitive=False), default="png", show_default=True)
@click.option("--every", type=int, default=1, show_default=True, help="Keep 1 of every N frames")
@click.option("--start", type=int, default=0, show_default=True, help="Start at frame index")
@click.option("--max-per-video", type=int, default=0, show_default=True, help="Cap per-video frames (0 = no cap)")
@click.option("--resize", type=int, default=0, show_default=True, help="Resize to NxN pixels (0 = keep original)")
@click.option("--gray/--no-gray", default=False, show_default=True, help="Save as grayscale")
@click.option("--keep-video-subdir/--flat", default=False, show_default=True, help="Create a subfolder per video")
@click.option("--manifest", type=click.Path(path_type=Path), default=Path("data/frames_manifest.csv"), show_default=True)
def extract_frames(input_dir, output_dir, ext, every, start, max_per_video, resize, gray, keep_video_subdir, manifest):
    """Extract frames from all .avi files and write a manifest CSV."""
    resize_val = int(resize) if int(resize) > 0 else None
    rows = extract_frames_from_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        img_ext=ext,
        every=every,
        start=start,
        max_per_video=max_per_video,
        resize=resize_val,
        gray=gray,
        keep_video_subdir=keep_video_subdir,
    )
    if not rows:
        raise click.ClickException("No frames extracted.")
    df = pd.DataFrame(rows)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest, index=False)
    click.echo(f"✅ Extracted {len(df)} frames")
    click.echo(f"📝 Manifest: {manifest}")
