# Overlay Reconstruction Pipeline

This note describes how we can regenerate coloured overlays for pseudo‑lateral CT slabs in two formats:

1. **From generator artefacts** – when `build-pseudo-lateral` has already written `mask_labels.png` (the multi-class ID map) alongside `image.png` and `mask.png`.
2. **From raw metadata only** – when only `image.png`, `mask.png`, and `labels.json` are available (e.g., model predictions for unseen data).

The second route is what the new `spine rebuild-overlay` command implements, and is the core we’ll reuse after a model produces its own labels.

---

## Inputs

Each slab directory produced by `build-pseudo-lateral` contains:

- `image.png` – grayscale projection (uint8).
- `mask.png` – union mask (white wherever any label is present).
- `labels.json` – metadata for each vertebra ID. For every label `L` it stores:
  - `present` flag.
  - `bbox_2d_index`: `(y0, x0, y1, x1)` bounding box in pixel indices.
  - `centroid_2d_index`: `(yc, xc)` centroid in pixel indices.
  - areas and other stats (not required for reconstruction).
- (Optionally) `mask_labels.png` – multi-class ID map. If present, we can colourise directly without JSON heuristics.

The reconstruction pipeline assumes at least `image.png`, `mask.png`, and `labels.json`. If `mask_labels.png` is present, the helper uses it unless you pass `--ignore-label-map`.

## Output Layout

Each angle directory now groups artefacts by type:

```
outputs/pseudo_lateral/angles/<rot_id>/images/<stem>.png
outputs/pseudo_lateral/angles/<rot_id>/mask_labels/<stem>.png
outputs/pseudo_lateral/angles/<rot_id>/mask/<stem>.png
outputs/pseudo_lateral/angles/<rot_id>/labels-json/<stem>.json
outputs/pseudo_lateral/angles/<rot_id>/overlays/<stem>.png
outputs/pseudo_lateral/angles/<rot_id>/overlay_recolored/<stem>.png
```

The manifest at `angles/<rot_id>/manifest.csv` now contains only two columns (`image`, `mask_labels`) pointing to the files above. All other metadata lives in the per-sample JSON.

---

## Reconstruction Steps

1. **Load artefacts:**
   - Read `image.png` as grayscale array `I` (shape `(H, W)`).
   - Read `mask.png` and threshold to a boolean union mask `U`.
   - Load label metadata from `labels.json`.

2. **Per-label region extraction:**
   For each label `L` marked `present`:
   - Take its bounding box `(y0, x0, y1, x1)` and crop the union mask to that ROI.
   - Remove pixels already assigned to earlier labels (processing order is sorted IDs; adjust if desired).
   - Run connected-component labelling inside the ROI and keep the largest component (guards against bleed from neighbouring vertebrae).
   - Mark those pixels in a label map `M` with the ID `L`. Also store the centroid `(yc, xc)` for later.

3. **Fill unassigned pixels:**
   - Some pixels in `U` may still be unassigned (overlaps, noise). For those, assign them to the nearest label centroid by Euclidean distance (works well because centroids are reliable anchor points).

4. **Colourisation:**
   - For each unique label ID in `M`, fetch the canonical BGR colour (matching other overlays).
   - Create a colour mask `C` (shape `(H, W, 3)`): `C[p] = colour(L)` where `M[p] = L`, else `[0,0,0]`.

5. **Blending:**
   - Convert `I` to BGR.
   - Blend `C` with `I` using weight `alpha` (default ≈ 0.45). Result is the recoloured overlay `O`.
   - Optionally, save the raw colour mask `C` for segmentation workflows.

This logic is implemented in `spine rebuild-overlay`:

```bash
poetry run spine rebuild-overlay \ 
  <sample_dir> \ 
  --out-path <output_png> \ 
  --mask-out <colour_mask_png> \ 
  --alpha 0.45
```

```bash
# sample commands for each reconstruction of the method

# generate a single file
bash scripts/build_pseudo_dataset.sh   --data-root data/CTSpine1K   --out-dir outputs/pseudo_lateral   --limit-cases 1   --override-existing --clear-dir --axis-helper --roll-sweep "0:0:0" --pitch-sweep "90:90:0" --yaw-sweep "0:0:0"

# generate the overlay using only the json metadata
poetry run spine rebuild-overlay \
  outputs/pseudo_lateral/angles/p+90_y+0_r+0/HNSCC-3DCT-RT_HN_P001.nii_slab00 \
  --ignore-label-map \
  --out-path outputs/pseudo_lateral/angles/p+90_y+0_r+0/HNSCC-3DCT-RT_HN_P001.nii_slab00/overlay_rebuilt.png \
  --mask-out outputs/pseudo_lateral/angles/p+90_y+0_r+0/HNSCC-3DCT-RT_HN_P001.nii_slab00/mask_color_rebuilt.png \
  --alpha 0.45
```

It only depends on the three base files, so it also works for model predictions that produce a `labels.json` and union mask.

---

## Integration with Future Model

For untrained images:

1. Run the inference model to generate:
   - `labels.json` (mirroring the generator schema: `present`, `bbox`, `centroid`, etc.).
   - `mask.png` (union mask of all predicted vertebrae).
   - `image.png` (original projection; the model will see it as input anyway).

2. Invoke `spine rebuild-overlay` on the prediction directory to create:
   - `overlay_rebuilt.png` – blended sandwich for visual QA / dataset curation.
   - `mask_color_rebuilt.png` (if `--mask-out` passed) – multi-class colour mask for downstream training or export.

For generator outputs that already include `mask_labels.png`, we can instead call `spine colorize-labels <sample_dir>` to colourise directly from the ID map; both commands rely on the same colour palette defined in `src/spine_segmentation/core/label_colors.py`.

These scripts are lightweight, avoid reloading the original NIfTI volumes, and ensure the overlay visualisations stay consistent whether they originate from synthetic projections or a trained model’s predictions.
