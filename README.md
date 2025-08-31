# INF2008

## requirements

- python version 3.12.7
- install ```pip install huggingface_hub nibabel```

## running with poetry

- ```poetry install --with dev```
- activate shell with ```poetry shell``` 

## commands

list subsets available at ctspine1k

```
spine download ctspine1k --list-subsets
```

## list 

download only head and neck scans

```
spine download ctspine1k --subset HNSCC-3DCT-RT --output-dir data/CTSpine1K
```

## Pseudo‑Lateral Dataset (Fluoro‑like)

Generate lateral‑style, fluoroscopy‑like images directly from CT volumes, with aligned masks and mm metadata. Useful to train a 2D segmenter that matches your larynx exam frames.

Example (orientation that matches your exam):

```
spine build-pseudo-lateral \
  --data-root data/CTSpine1K \
  --subset HNSCC-3DCT-RT \
  --limit-cases 1 \
  --labels 1-7 \
  --plane sag \
  --yaw 0 --pitch 90 --roll -90 \
  --rot90 1 \
  --aperture \
  --slab-mm 12 \
  --slab-count 30 \
  --out-dir data/pseudo_lateral
```

- plane: projection plane. `sag` projects along X (left‑right) → lateral profile.
- yaw/pitch/roll: 3D rotations (deg) applied before projection: Z/Y/X respectively.
- rot90/flip-h/flip-v: 2D orientation tweaks after projection so the overlay matches your frame.
- aperture: draws a circular c‑arm vignette and slight boost inside the field.
- slab-mm / slab-count: generates multiple depth “slabs” in millimeters along the projection axis; centered around mid‑depth. Slabs simulate occlusions/partials. Use `--slab-step-mm` to control overlap (default = non‑overlap).

Outputs written under `data/pseudo_lateral`:
- `images/<subset_case>_slabXX.png`: pseudo‑fluoro image for slab XX
- `masks/<subset_case>_slabXX.png`: union silhouette of requested labels (e.g., C1..C7)
- `overlays/<subset_case>_slabXX.png`: quick visual overlay
- `<subset_case>_slabXX.json`: metadata with mm spacing and per‑label stats

JSON contents (per slab):
- `pixel_spacing`: `[row_mm, col_mm]`
- `proj_axis_spacing_mm`: spacing along projection axis (mm/voxel)
- `slab_vox`, `slab_mm`, `slab_step_vox`, `slab_step_mm`
- `slab_range_index` and `slab_range_mm`: voxel and mm bounds of the slab inside the rotated volume
- Per‑label (`labels`): `present`, `centroid_2d_index`, `centroid_2d_mm`, `bbox_2d_index`, `bbox_2d_hw_mm`, `area_px`, `area_mm2`

Tip: with the example above, slabs around index ~14 often look very similar to lateral larynx fluoroscopy frames.

## Compare To ImageJ Masks

Compare a frame with its ImageJ mask (and optionally another mask) and compute IoU/Dice.

```
spine compare-masks \
  --frame data/frames/50/v50_f145.png \
  --manual /path/to/Mask.tif \
  --other outputs/register_debug/95_projected_mask.png \
  --out-dir outputs/compare_manual
```

## Optional: DRR Bank + Registration (coarse-to-fine)

Build a descriptor bank from CTSpine1K and register a lateral frame for a projected overlay.

```
spine build-bank \
  --data-root data/CTSpine1K \
  --subset HNSCC-3DCT-RT \
  --limit-cases 8 \
  --views-per-case 24 \
  --height 512 \
  --auto-fov \
  --masked \
  --out data/banks/hnscc_bank.npz \
  --debug-dir outputs/bank_debug \
  --unpack-dir data/banks/hnscc_bank_unpacked

spine register \
  --bank data/banks/hnscc_bank.npz \
  --frame data/frames/50/v50_f145.png \
  --iters 250 \
  --topk 5 \
  --out outputs/overlay.png \
  --debug-dir outputs/register_debug
```

Notes:
- Banks created with `--auto-fov` are handled downstream (delx recomputed as needed).
- Use `spine scan-labels` to discover available label IDs per subset.

## Frames: Extract and Index

Extract frames from `.avi` and build a manifest.

```
spine extract-frames \
  --input-dir data/videos \
  --output-dir data/frames \
  --ext png --every 1 --keep-video-subdir \
  --manifest data/frames_manifest.csv

spine frames-manifest \
  --frames-dir data/frames \
  --write data/frames_metadata.csv \
  --masks-root /path/with/Mask.tif
```

## Troubleshooting
- If DRR routes produce black images, prefer the pseudo‑lateral generator above (DRR‑free), or build banks with `--no-masked` and `--auto-fov`.
- For pseudo‑laterals, adjust `--plane`, `--yaw/--pitch/--roll`, and `--rot90/--flip-h/--flip-v` until overlays align with your fluoroscopy.

build a bank of DRRs (masked to vertebrae) for coarse matching

```
spine build-bank \
  --data-root data/CTSpine1K \
  --subset HNSCC-3DCT-RT \
  --limit-cases 8 \
  --views-per-case 24 \
  --height 512 \
  --auto-fov \
  --masked \
  --out data/banks/hnscc_bank.npz \
  --debug-dir outputs/bank_debug \
  --unpack-dir data/banks/hnscc_bank_unpacked
```

register a lateral frame against the bank and overlay the projected mask

```
spine register \
  --bank data/banks/hnscc_bank.npz \
  --frame data/frames/50/v50_f145.png \
  --iters 250 \
  --topk 5 \
  --out outputs/overlay.png \
  --debug-dir outputs/register_debug
```

compare a frame with its ImageJ mask (and optionally another mask) and compute IoU/Dice

```
spine compare-masks \
  --frame data/frames/50/v50_f145.png \
  --manual /path/to/Mask.tif \
  --other outputs/register_debug/95_projected_mask.png \
  --out-dir outputs/compare_manual
```

build a synthetic 2D dataset (DRR + projected mask) for training a convnet

```
spine build-synth \
  --data-root data/CTSpine1K \
  --subset HNSCC-3DCT-RT \
  --limit-cases 30 \
  --views-per-case 30 \
  --height 512 \
  --auto-fov \
  --masked \
  --out-dir data/synth2d
```

train a simple UNet for 2D spine mask segmentation using the synthetic dataset

```
spine train-seg2d --dataset-dir data/synth2d --epochs 30 --batch 4 --out-dir outputs/seg2d
```

predict masks over frames using the trained UNet

```
spine predict-seg2d \
  --model outputs/seg2d/model.pt \
  --frame data/frames/50/v50_f145.png \
  --out-dir outputs/seg2d_pred
```

Then compare predicted masks with ImageJ using `spine compare-masks`.

build pseudo-lateral images (fluoroscopy-like) from CT volumes for training

```
spine build-pseudo-lateral \
  --data-root data/CTSpine1K \
  --subset HNSCC-3DCT-RT \
  --limit-cases 8 \
  --labels 1-7 \
  --height 512 \
  --aperture \
  --out-dir data/pseudo_lateral
```

This creates `images/` (pseudo X-ray), `masks/` (C1..C7 union silhouettes), overlays, and a per-case JSON with 2D centroids/bboxes.

Notes:
- Banks created with `--auto-fov` are supported end-to-end; delx is recomputed on-the-fly during export and registration.
- If you pass `--labels` to `build-bank` but a case has no segmentation file, that case is skipped (to respect the label filter).
