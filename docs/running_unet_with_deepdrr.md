# UNet Training with DeepDRR-Style Inputs

This page captures the exact commands to reproduce the new DeepDRR-informed
pipeline. It starts from a clean checkout and ends with batch inference over
VFSS frames, so you can follow it step-by-step for the upcoming presentation.

> ⚠️ The legacy `build-pseudo-lateral` generator has been deprecated. Use
> `spine build-hf-projection` (or the DeepDRR bridge below) when the guide asks
> for projection inputs.

## 1. Environment setup

```bash
poetry install
poetry run pip install deepdrr[cuda12x]
# (already vendored) cp third_party/deepdrr/lib/linux/libXrender.so.1* if needed
```

> Why deepdrr? See `docs/deepdrr_integration.md` for the physics rationale.

## 2. Single-image sanity check

Render one fluoroscopy-like view straight from the CT volume:

```bash
PYOPENGL_PLATFORM=egl \
PYGLET_HEADLESS=true \
poetry run spine deepdrr-project \
   --ct data/CTSpine1K/raw_data/volumes/HNSCC-3DCT-RT/HN_P001.nii.gz \
   --out outputs/deepdrr/HNSCC-3DCT-RT_HN_P001_pitch0.png \
   --pitch 180 --yaw 0 --roll 0 
```

Inspect `outputs/deepdrr/HNSCC-3DCT-RT_HN_P001_pitch0.png` and compare it to a
high-fidelity projection (`outputs/high_fidelity/...`). The new image should
look softer and closer to fluoroscopy tone.

### Before/after snapshots

```bash
# Raw (before tone mapping)
PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true \
poetry run spine deepdrr-project \
  --ct data/CTSpine1K/raw_data/volumes/HNSCC-3DCT-RT/HN_P001.nii.gz \
  --out outputs/deepdrr/HNSCC-3DCT-RT_HN_P001_pitch0_raw.png \
  --pitch 180 --tone raw

# Tone-mapped (default smooth gamma)
PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true \
poetry run spine deepdrr-project \
  --ct data/CTSpine1K/raw_data/volumes/HNSCC-3DCT-RT/HN_P001.nii.gz \
  --out outputs/deepdrr/HNSCC-3DCT-RT_HN_P001_pitch0_smooth.png \
  --pitch 180

# Tone-mapped + CLAHE
PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true \
poetry run spine deepdrr-project \
  --ct data/CTSpine1K/raw_data/volumes/HNSCC-3DCT-RT/HN_P001.nii.gz \
  --out outputs/deepdrr/HNSCC-3DCT-RT_HN_P001_pitch0_clahe.png \
  --pitch 180 --clahe
```

Open the three PNGs side-by-side to judge the impact of tone mapping and CLAHE
on local contrast.

## 3. Rebuild training tiles with DeepDRR renderer

(WIP) Once `prepare-unet-tiles` exposes a `--renderer deepdrr` option, run:

```bash
poetry run spine prepare-unet-tiles \
  --csv outputs/high_fidelity/manifest.csv \
  --out-dir prepared/unet_tiles_deepdrr/c4 \
  --renderer deepdrr \
  --height 512 \
  --pitch 90 --yaw 0 --roll 0
```

For now, generate a pilot set by looping `deepdrr-project` over the slabs you
want to replace and stash the outputs under `prepared/unet_tiles_deepdrr/...`.

## 4. Train UNet on the new tiles

```bash
poetry run spine train-unet \
  --csv prepared/unet_tiles_deepdrr/c4/manifest.csv \
  --out-dir runs/unet_deepdrr_exp01 \
  --epochs 40 \
  --batch-size 6 \
  --learning-rate 5e-4 \
  --val-fraction 0.15 \
  --test-fraction 0.1 \
  --patience 8 \
  --monitor dice \
  --augment \
  --amp
```

Track `runs/unet_deepdrr_exp01/metrics.csv` and `training_summary.json` to see
how the new training data impacts Dice/IoU.

## 5. Batch inference on VFSS frames

```bash
poetry run spine predict-unet-batch \
  --model runs/unet_deepdrr_exp01/unet_best.pt \
  --image-dir data/frames/50 \
  --pattern "*.png" \
  --overlay-dir runs/inference/deepdrr_overlays \
  --mask-dir runs/inference/deepdrr_masks \
  --summary-csv runs/inference/deepdrr_summary.csv \
  --clahe
```

Open `runs/inference/deepdrr_summary.csv` and highlight:

- fraction of frames with non-zero detections,
- pixel counts per vertebra label,
- overlays (check the `_clahe` helper images) for qualitative improvements.

## 6. Presentation checklist

- Before/after comparison: high-fidelity (CPU) vs DeepDRR on the same slab.
- Training curves and metric table from `runs/unet_deepdrr_exp01`.
- Batch summary statistics (how many VFSS frames gained meaningful detections).
- Notes on remaining gaps (e.g. no scatter yet, still need to integrate the
  official DeepDRR CNNs, fine-tuning on real VFSS labels).

With these artefacts in hand you can tell the story end-to-end: physics-grounded
simulator → new training data → improved inference behaviour.
