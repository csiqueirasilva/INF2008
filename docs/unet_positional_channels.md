# UNet Positional + Foreground Channels (4-Channel Input)

This variant feeds the UNet more spatial priors so it can stay anchored when the neck crop shifts:

- **Channel 1:** `clahe2` (grayscale, already contrast-enhanced in the dataset).
- **Channel 2:** Otsu neck mask (computed on `clahe2` with Gaussian blur `k=5`).
- **Channels 3–4:** Normalized `y` and `x` coordinate grids in `[0,1]`.

The model expects these 4 channels at both training and inference time. A shape mismatch will raise an error if the flags don’t match the checkpoint.

## Train
Use the new flags in `train-unet`:

```bash
poetry run spine train-unet \
  --csv prepared/unet_cropped6_clahe/manifest.csv \
  --out-dir runs/unet_cropped6_clahe_pos \
  --epochs 40 --batch-size 6 --learning-rate 5e-4 \
  --val-fraction 0.15 --test-fraction 0.10 \
  --monitor dice --patience 8 --amp --augment \
  --use-otsu-channel --use-coord-channels --otsu-blur-kernel 5
```

The checkpoint stores `num_input_channels`, `use_otsu_channel`, `use_coord_channels`, and `otsu_blur_kernel` so inference can auto-match the same layout.

## Inference (batch)
When predicting, keep the channel flags consistent (defaults read from the checkpoint):

```bash
poetry run spine predict-unet-batch \
  --model runs/unet_cropped6_clahe_pos/unet_best.pt \
  --image-dir outputs/frames7_crops_clahe/crops \
  --pattern "*_crop.png" \
  --overlay-dir outputs/frames7_crops_clahe/unet_overlays_pos \
  --mask-dir outputs/frames7_crops_clahe/unet_masks_pos \
  --no-clahe \
  --use-otsu-channel --use-coord-channels --otsu-blur-kernel 5
```

For single-image inference, the same flags apply to `predict-unet`.

## Notes
- The 1-channel (clahe2-only) model underperformed on real frames (labels drifted, class separation was poor). Adding the Otsu foreground prior + coord channels significantly improved alignment on out-of-domain frames (e.g., vid7 crops) because the model now has both a neck mask and positional cues.
- If you change the Otsu blur kernel or disable a channel, the command will refuse to run when the resolved channel count does not match the checkpoint.
- Augmentations now leave the coordinate channels untouched (only the first image channel gets intensity/noise jitter). Spatial transforms (flip/rotation) still apply to all channels.***
