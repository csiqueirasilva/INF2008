from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import nibabel as nib
import click

def load_nii(path: Path):
    nii = nib.load(str(path))
    arr = np.asarray(nii.get_fdata(), dtype=np.float32)
    spacing = np.array(nii.header.get_zooms()[:3], dtype=np.float32)
    origin = np.array(nii.affine[:3, 3], dtype=np.float32)
    return arr, spacing, origin

def _first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None

def _base_from_nii_like(p: Path) -> str:
    """
    Return filename without NIfTI suffix, handling .nii.gz properly.
    e.g., HN_P001.nii.gz -> HN_P001
          HN_P001.nii    -> HN_P001
    """
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem

def _find_matching_label(lab_dir: Optional[Path], stem: str) -> Optional[Path]:
    if lab_dir is None or not lab_dir.exists():
        return None
    # Preferred CTSpine1K pattern
    cand = lab_dir / f"{stem}_seg.nii.gz"
    if cand.exists():
        return cand
    # Tolerant fallbacks
    for pat in (f"{stem}*seg*.nii*", f"{stem}*label*.nii*", f"{stem}*labels*.nii*"):
        hits = sorted(lab_dir.glob(pat))
        if hits:
            return hits[0]
    return None

def find_pairs(data_root: Path, subset: str, limit: int = 0) -> List[Tuple[Path, Optional[Path]]]:
    """
    Works with either:
      <data_root>/raw_data/{volumes,labels}/<subset>/
      <data_root>/{volumes,labels}/<subset>/
    Returns (ct_path, seg_path or None).
    """
    vol_dir = _first_existing([
        data_root / "raw_data" / "volumes" / subset,
        data_root / "volumes"  / subset,
    ])
    lab_dir = _first_existing([
        data_root / "raw_data" / "labels"  / subset,
        data_root / "labels"    / subset,
    ])

    if vol_dir is None:
        raise click.ClickException(
            f"Volumes dir not found. Looked under:\n"
            f"  {data_root/'raw_data'/'volumes'/subset}\n"
            f"  {data_root/'volumes'/subset}"
        )

    vols = sorted(vol_dir.glob("*.nii*"))
    if not vols:
        raise click.ClickException(f"No volumes (*.nii*) in {vol_dir}")

    pairs: List[Tuple[Path, Optional[Path]]] = []
    for v in vols:
        base = _base_from_nii_like(v)        # ← FIX: strip .nii(.gz) properly
        seg  = _find_matching_label(lab_dir, base)
        pairs.append((v, seg))
        if limit and len(pairs) >= limit:
            break
    return pairs