# Segment Detection Pipeline (UNet-Based)

> ⚠️ This draft targeted the legacy pseudo-lateral dataset. Replace the paths
> below with the outputs from `spine build-hf-projection` when following the
> current workflow.

This document outlines the planned workflow for detecting vertebral segments in pseudo-lateral projections using a UNet-style network.

## Data Representation

- Input: `outputs/high_fidelity/images/<stem>.png`
- Target: `outputs/high_fidelity/mask_labels/<stem>.png`
- Optional diagnostics: union mask, overlays, and `labels-json/<stem>.json`.

## Dataset Layout

Training pairs live in the projection folder created during generation, e.g. `outputs/high_fidelity/images/<stem>.png` alongside the matching `mask_labels/<stem>.png`. The two columns in `manifest.csv` point to those files directly.

## Training Pipeline

1. Use `image.png` as input to the network (single-channel or replicate to 3 channels as needed).
2. Use `mask_labels.png` as the supervision target. Train a UNet (or similar encoder-decoder) with multi-class segmentation loss (cross-entropy, Dice, etc.).
3. Optionally, augment with intensity/geometry variations consistent with the synthetic generation parameters.

## Inference and Reconstruction

1. Run the trained UNet on a new `image.png` to obtain per-pixel logits.
2. Take an argmax over the label dimension to produce a predicted ID map (`mask_labels_pred`).
3. Feed the predicted map to the colouring helper (e.g. `spine colorize-labels`) to produce `overlay_unet_recolored.png` (visual inspection) and `mask_color.png` (multi-class colour mask).

This approach keeps the network’s output fully compatible with the existing colourisation and overlay tooling and avoids lossy heuristics (bounding boxes, centroids) during reconstruction.
