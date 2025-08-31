import click, json
from pathlib import Path
import numpy as np
from .root import cli
from ..core.io import find_pairs, load_nii

@cli.command("scan-labels")
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/CTSpine1K"), show_default=True)
@click.option("--subset", default="HNSCC-3DCT-RT", show_default=True)
@click.option("--limit-cases", type=int, default=0, show_default=True, help="0 = all")
def scan_labels(data_root, subset, limit_cases):
    """Print unique label IDs and counts across a subset."""
    pairs = find_pairs(data_root, subset, limit=None if limit_cases==0 else limit_cases)
    counts = {}
    for _, seg in pairs:
        lab, _, _ = load_nii(seg)
        vals, cnt = np.unique(lab.astype(np.int32), return_counts=True)
        for v, c in zip(vals, cnt):
            counts[int(v)] = counts.get(int(v), 0) + int(c)
    # pretty print
    ids = sorted(counts.keys())
    click.echo(f"Subset: {subset}")
    click.echo(f"Labels present: {ids}")
    click.echo(json.dumps({str(k): counts[k] for k in ids}, indent=2))
