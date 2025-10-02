# Segment Detection Pipeline (UNet-Based)

This document outlines the planned workflow for detecting vertebral segments in pseudo-lateral projections using a UNet-style network.

## Data Representation

Each projected slab directory contains:

- `image.png` – grayscale pseudo-lateral view (uint8).
- `mask_labels.png` – multi-class label map (uint8) where each pixel stores the vertebra ID (0 = background, 1–7 for C1–C7).
- `mask.png` – union mask (any label > 0), provided mainly for reference.
- `labels.json` – metadata (centroids, bounding boxes, statistics) for each label; useful for QA but not required for the neural net.

## Training Pipeline

1. Use `image.png` as input to the network (single-channel or replicate to 3 channels as needed).
2. Use `mask_labels.png` as the supervision target. Train a UNet (or similar encoder-decoder) with multi-class segmentation loss (cross-entropy, Dice, etc.).
3. Optionally, augment with intensity/geometry variations consistent with the synthetic generation parameters.

## Inference and Reconstruction

1. Run the trained UNet on a new `image.png` to obtain per-pixel logits.
2. Take an argmax over the label dimension to produce a predicted ID map (`mask_labels_pred`).
3. Feed the predicted map to the colouring helper (e.g. `spine colorize-labels`) to produce `overlay_unet_recolored.png` (visual inspection) and `mask_color.png` (multi-class colour mask).

This approach keeps the network’s output fully compatible with the existing colourisation and overlay tooling and avoids lossy heuristics (bounding boxes, centroids) during reconstruction.
