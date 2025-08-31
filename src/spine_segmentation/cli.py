# src/spine_segmentation/cli.py
from .commands.root import cli 

from .commands import (
    download, build_bank, register, extract_frames, frames_manifest,
    inspect, scan_labels, compare_masks, build_synth, train_seg2d, predict_seg2d, mip_labels, build_pseudo_lateral
)

if __name__ == "__main__":
    cli()
