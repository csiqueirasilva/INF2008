# Preparing UNet Training Tiles

`spine prepare-unet-tiles` crops circular 384×384 tiles centred on the centroid of a chosen vertebra (default C4) and saves them with matching label maps. This produces UNet-ready inputs while mimicking the framing in `data/frames/`.

## Command

```bash
poetry run spine prepare-unet-tiles \
  --csv outputs/pseudo_lateral/angles/p+90_y+0_r+0/manifest.csv \
  --out-dir prepared/unet_tiles/c4 \
  --size 384 \
  --label-id 4 \
  --radius 192
```

```bash
# or omit label-id to use averaged centroid \
poetry run spine prepare-unet-tiles \
  --csv outputs/pseudo_lateral/angles/p+90_y+0_r+0/manifest.csv \
  --out-dir prepared/unet_tiles/c4 \
  --size 384 \
  --radius 192
```

### Important options

| Flag | Description |
|------|-------------|
| `--csv` | Source manifest (only `image` + `mask_labels` columns required). |
| `--out-dir` | Output root (folders `images/`, `mask_labels/`, `overlays/`, `manifest.csv`). |
| `--size` | Tile size (square). Defaults to 384. |
| `--label-id` | Specific vertebra ID to centre on. Omit to use the average centroid over all present labels. |
| `--radius` | Circular vignette radius in pixels (defaults to size/2). Pixels outside are zeroed. |
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

1. Generate pseudo-lateral projections with `scripts/build_pseudo_dataset.sh`.
2. Run `spine prepare-unet-tiles` to build the centered, circular tiles.
3. Train with `spine train-unet --csv <out_dir>/manifest.csv ...`.

This keeps training data aligned with the future inference pipeline, which will also emit tiles centred on C4 (or another chosen label) before running the UNet.
