# Pseudo-Lateral Dataset Generation

`build_pseudo_dataset.sh` is a thin wrapper around `spine build-pseudo-lateral`. It sweeps yaw/pitch/roll angles, renders pseudo-lateral projections, and writes each slab into the structured layout under `outputs/pseudo_lateral/angles/<rot_id>/...`.

## Directory layout

For every angle identifier (e.g. `p+90_y+0_r+0`) we now emit:

```
angles/<rot_id>/images/<stem>.png
angles/<rot_id>/mask_labels/<stem>.png
angles/<rot_id>/mask/<stem>.png
angles/<rot_id>/labels-json/<stem>.json
angles/<rot_id>/overlays/<stem>.png
angles/<rot_id>/overlay_recolored/<stem>.png
angles/<rot_id>/manifest.csv  # two columns: image, mask_labels
```

`<stem>` is always `<subset>_<case>_slabXX` so slabs stay grouped together and easy to inspect.

## Slab sampling parameters

Depth sampling is controlled by four switches:

| Parameter      | Default | Meaning |
|----------------|---------|---------|
| `SLAB_MM`      | 12      | Thickness of each slab along the projection axis.
| `KEEP_MM`      | 0       | Half-window around the mid-plane (in millimetres). If > 0 it limits the slab centres to ±`KEEP_MM` from the centre slice.
| `WINDOW_COUNT` | 1       | Number of slab centres to sample **inside the ±`KEEP_MM` window**. Only used when `KEEP_MM > 0`.
| `SLAB_COUNT`   | 1       | Number of slabs when no window is specified (`KEEP_MM == 0` or `WINDOW_COUNT == 0`).

### Why did `WINDOW_COUNT=50 KEEP_MM=30 SLAB_COUNT=50` produce only ~30 slabs?

For the HNSCC volumes the spacing along the projection axis is 2 mm. With `KEEP_MM=30` the window spans `±15` voxels (`30 / 2 = 15`). When we request `WINDOW_COUNT=50`, the code generates 50 evenly-spaced centres **inside that 30-voxel window**, but after rounding centres to integer voxels and deduplicating there can be at most 31 unique centres (`2 × 15 + 1`). In practice we observed 30 unique slabs because different floating-point samples rounded to the same integer index.

`SLAB_COUNT` is ignored in that scenario because the `KEEP_MM`/`WINDOW_COUNT` branch is active.

### How do I truly get 50 slabs within ±30 mm?

You cannot while the spacing is 2 mm: there are only 31 distinct voxel centres to choose from inside ±30 mm. Options:

1. Increase `KEEP_MM` (e.g. 60 mm) so the window covers more voxels; `WINDOW_COUNT` may then reach 50 unique centres.
2. Reduce `SLAB_MM` (thinner slabs) and widen `KEEP_MM` if you need more overlap but still want to stay within a physical window.
3. Set `KEEP_MM=0` and use `SLAB_COUNT=50` to force evenly-spaced slabs across the entire volume (`WINDOW_COUNT` will be ignored).

### Summary of branch logic

```
if KEEP_MM > 0 and WINDOW_COUNT > 0:
    # sample WINDOW_COUNT centres within the ±KEEP_MM window (duplicates removed)
else:
    # fall back to symmetric sampling with SLAB_COUNT slabs around the mid-plane
```

Remember that the number of **unique** slabs is ultimately limited by the discrete voxel spacing and any rounding that occurs during centre selection.

## Running the script

Examples:

```bash
# single angle, default sampling
bash scripts/build_pseudo_dataset.sh \
  --data-root data/CTSpine1K \
  --out-dir outputs/pseudo_lateral \
  --limit-cases 1 \
  --override-existing --clear-dir --axis-helper \
  --roll-sweep "0:0:0" --pitch-sweep "0:0:0" --yaw-sweep "0:0:0"

# dense sweep with ±30 mm window and 31 slabs
KEEP_MM=30 WINDOW_COUNT=50 SLAB_MM=12 bash scripts/build_pseudo_dataset.sh ...
# (actual unique slabs <= 31 because spacing is 2 mm)

# symmetric sampling across the entire volume
KEEP_MM=0 SLAB_COUNT=50 bash scripts/build_pseudo_dataset.sh ...
```

Use the generated folders to visually inspect slabs (`angles/<rot_id>/images`), masks, and overlays while calibrating the depth parameters.

# sample command to generate for all head and neck cases, lateral view

```bash
KEEP_MM=60 WINDOW_COUNT=50 SLAB_MM=12 bash scripts/build_pseudo_dataset.sh   --data-root data/CTSpine1K   --out-dir outputs/pseudo_lateral   --limit-cases 50   --override-existing --clear-dir --axis-helper --roll-sweep "0:0:0" --pitch-sweep "0:0:0" --yaw-sweep "0:0:0"
```
## Preparing centred tiles

Use `spine prepare-unet-tiles` to crop circular 384×384 tiles (centred on C4 by default). The command reads any angle manifest and writes a new slim manifest under the output directory.
