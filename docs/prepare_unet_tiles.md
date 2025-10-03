# Preparing UNet Training Tiles

`spine prepare-unet-tiles` crops circular 384×384 tiles centred on the centroid of a chosen vertebra (default C4) and saves them with matching label maps. This produces UNet-ready inputs while mimicking the framing in `data/frames/`.

## Command

```bash
poetry run spine prepare-unet-tiles \
  --csv outputs/high_fidelity/manifest.csv \
  --out-dir prepared/unet_tiles/c4 \
  --size 384 \
  --label-id 4 \
  --radius 192
```

```bash
# or omit label-id to use averaged centroid \
poetry run spine prepare-unet-tiles \
  --csv outputs/high_fidelity/manifest.csv \
  --out-dir prepared/unet_tiles/c4 \
  --size 384 \
  --radius 192
```

### Single-sample inspection

When you only need to sanity-check one projection/mask pair, skip the manifest:

```bash
poetry run spine prepare-unet-tiles \
  --image outputs/high_fidelity/images/HN_P001.nii.png \
  --mask outputs/high_fidelity/mask_labels/HN_P001.nii.png \
  --annotation outputs/high_fidelity/labels-json/HN_P001.nii.json \
  --out-dir outputs/high_fidelity/fake-vfss \
  --label-id 4 \
  --zoom 1.25

# DeepDRR variant reusing the same mask/annotation
poetry run spine prepare-unet-tiles \
  --image outputs/deepdrr/HNSCC-3DCT-RT_HN_P001_pitch0.png \
  --mask outputs/high_fidelity/mask_labels/HN_P001.nii.png \
  --annotation outputs/high_fidelity/labels-json/HN_P001.nii.json \
  --out-dir outputs/deepdrr/fake-vfss \
  --label-id 4 \
  --zoom 1.25
```

`prepare-unet-tiles` will derive the matching `labels-json/...` automatically (or
accept `--annotation` if you want to point at it explicitly).

### Native-resolution projections

Use the new `--preserve-aspect` and `--native-resolution` flags to keep the
pixel grid emitted by the projection stages aligned with the source CT. This
removes the implicit 512×512 square resize so tile crops reflect the anatomical
aspect ratio you trained on.

Examples:

```bash
# High-fidelity DRR without forcing a square resize
poetry run spine build-hf-projection \
  --data-root data/CTSpine1K \
  --subset HNSCC-3DCT-RT \
  --yaw 0 --pitch 90 --roll 0 \
  --preserve-aspect \
  --out-dir outputs/high_fidelity_native \
  --limit-cases 1

# DeepDRR inspection frame using the CT-derived pixel grid
poetry run spine deepdrr-project \
  --ct data/CTSpine1K/HNSCC-3DCT-RT/HN_P001.nii \
  --out outputs/deepdrr/HN_P001_native.png \
  --pitch 90 \
  --native-resolution

# Legacy pseudo-lateral support has been deprecated; use build-hf-projection instead.
```

When you still need fixed-size images for batching but want to avoid the square
stretch, pair `--height` with `--preserve-aspect` (for example, 768×432). The
projection metadata in `labels-json/` captures the updated pixel spacing, so the
tile prep step keeps centroids and millimetre measurements consistent.

### Important options

| Flag | Description |
|------|-------------|
| `--csv` | Source manifest (only `image` + `mask_labels` columns required). |
| `--image`/`--mask` | Direct projection + mask paths (repeatable; bypasses `--csv`). |
| `--annotation` | Optional `labels-json` path when supplying `--image`/`--mask`. |
| `--out-dir` | Output root (folders `images/`, `mask_labels/`, `overlays/`, `manifest.csv`). |
| `--size` | Tile size (square). Defaults to 384. |
| `--label-id` | Specific vertebra ID to centre on. Omit to use the average centroid over all present labels. |
| `--radius` | Circular vignette radius in pixels (defaults to size/2). Pixels outside are zeroed. |
| `--zoom` | Zoom factor (>1 crops tighter then resizes back to output size). |
| `--write-overlay` | Enable/disable QA overlays. |

## Output structure

```
out_dir/
  images/<stem>.png            # 384×384 grayscale tiles
  mask_labels/<stem>.png       # matching multi-class mask
  overlays/<stem>.png          # optional recoloured overlay for inspection
  manifest.csv                 # two columns (image, mask_labels)
```

Tiles are cropped around the centroid retrieved from the corresponding `labels-json/<stem>.json`. Samples without the requested label or centroid are skipped (reported in the final summary).

## Workflow

1. Generate projections with `spine build-hf-projection` (or the DeepDRR bridge).
2. Run `spine prepare-unet-tiles` to build the centered, circular tiles.
3. Train with `spine train-unet --csv <out_dir>/manifest.csv ...`.

This keeps training data aligned with the future inference pipeline, which will also emit tiles centred on C4 (or another chosen label) before running the UNet.
