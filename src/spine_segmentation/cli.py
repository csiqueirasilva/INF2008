# src/spine_segmentation/cli.py
from .commands.root import cli 

from .commands import download, build_bank, register, extract_frames, frames_manifest, inspect, scan_labels

if __name__ == "__main__":
    cli()
