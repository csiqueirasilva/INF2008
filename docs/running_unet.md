# Running UNet Inference

The `spine predict-unet` command loads a trained UNet checkpoint, runs segmentation on a single image, and produces a colour overlay using the same palette as the dataset generator. Optionally it can also persist the raw `mask_labels.png` emitted by the network.

## Requirements

- A checkpoint produced by `spine train-unet` (for example `runs/unet_exp01/unet_best.pt`).
- A grayscale inference image that matches the tile dimensions used during training.
- The output directory must be writable so the command can store the overlay and, if requested, the predicted mask.

## Example

```bash
poetry run spine predict-unet \
  --model runs/unet_exp01/unet_best.pt \
  --image data/frames/50/v50_f100.png \
  --overlay-out runs/inference/v50_f100_overlay.png \
  --mask-out runs/inference/v50_f100_mask.png
```

The command executes the following steps:

- Load the model weights and class count from the checkpoint.
- Run the UNet forward pass on the provided image (CUDA with automatic fallback to CPU).
- Save the predicted label IDs as a PNG (`--mask-out` path). If `--mask-out` is omitted the mask is written to a temporary file that is deleted after overlay generation.
- Call `spine rebuild-overlay` internally to blend the grayscale image with the predicted mask, writing the result to `--overlay-out`.
- Report how many pixels were assigned to each predicted label so you can quickly sanity-check detections from the CLI output.

### Sanity check with a training tile

To confirm the model is behaving as expected, run the command against one of the tiles it was trained on:

```bash
# slab 22 was chosen due to the clear visible spine segments
poetry run spine predict-unet \
  --model runs/unet_exp01/unet_best.pt \
  --image prepared/unet_tiles/c4/images/HNSCC-3DCT-RT_HN_P001.nii_slab22.png \
  --overlay-out runs/inference/HNSCC-3DCT-RT_HN_P001.nii_slab22_overlay.png \
  --mask-out runs/inference/HNSCC-3DCT-RT_HN_P001.nii_slab22_mask.png
```

You should see a `🔎 Detected labels -> ...` line listing C1–C7 with small but non-zero pixel counts. If this sanity check fails, revisit the training dataset paths or checkpoint before debugging new inference inputs (video frames have a larger domain shift than the training tiles).

### Batch-processing a directory

When you want to sweep through an entire folder (for example, `data/frames/50`), use the batch command. It loads the model once, iterates over all files matching the glob pattern, and optionally writes a CSV summary.

> Looking for more realistic training projections? See `docs/deepdrr_integration.md` for the DeepDRR-inspired renderer that generates fluoroscopy-like inputs before UNet training.

```bash
poetry run spine predict-unet-batch \
  --model runs/unet_exp01/unet_best.pt \
  --image-dir data/frames/50 \
  --pattern "*.png" \
  --overlay-dir runs/inference/batch_overlays \
  --mask-dir runs/inference/batch_masks \
  --summary-csv runs/inference/batch_summary.csv \
  --clahe
```

The CLI will print per-image detection summaries and finish with a count of how many frames produced foreground labels. The CSV contains one row per image with pixel counts and percentages for each detected label.

## Options

| Flag | Description |
|------|-------------|
| `--model` | Path to the `.pt` checkpoint saved by `train-unet`. Required. |
| `--image` | Grayscale PNG (or compatible) to segment. Required. |
| `--overlay-out` | Destination PNG for the colour overlay. Required. |
| `--mask-out` | Optional path to store the raw prediction (`uint8` label map). If omitted the mask is transient. |
| `--device` | Torch device string (defaults to `cuda` when available, otherwise `cpu`). |
| `--alpha` | Blend factor passed to `rebuild-overlay`; higher values emphasise colour (default: 0.45). |
| `--clahe/--no-clahe` | Apply CLAHE contrast enhancement to the input before inference (default: disabled). |
| `--pattern` (batch) | Glob applied within `--image-dir` (defaults to `*.png`). |
| `--overlay-dir`, `--mask-dir` (batch) | Destination folders for per-image overlays and masks. |
| `--summary-csv` (batch) | Optional CSV capturing detections for the whole run. |

All paths accept `~` expansion and will be created as needed (for example, the parent directory of `--overlay-out`).

## Inspecting Results

- Use any image viewer to inspect `--overlay-out`. The legend and colour scheme match the high-fidelity projection dataset produced by `build-hf-projection`.
- Review the CLI summary (`🔎 Detected labels -> ...`) to confirm that vertebrae were found and to gauge their relative area.
- Try `--clahe` on low-contrast images; the equalisation often helps when moving from CT-style tiles to camera captures. When CLAHE is enabled, the preprocessed grayscale is also saved alongside the overlay (`*_clahe.png`).
- The UNet predicts each class independently via argmax of the logits, so single classes can appear without their neighbours if the network is confident; the batch report highlights such sparsely detected labels.
- Enable `--mask-out` when you want to compare predictions across runs, compute additional metrics, or feed the mask into downstream tooling.
- Because the command reuses `rebuild-overlay`, you can later re-run that command directly if you tweak the mask (e.g. after manual edits) and wish to regenerate the overlay.

## Troubleshooting

- **Shape mismatch** – Ensure the inference image has the same height/width as the training tiles; U-Net requires dimensions divisible by 16.
- **Missing file** – Confirm the path passed to `--image` exists (e.g. `data/frames/50/v50_f100.png` in this repository) and is readable.
- **No detections** – Verify the sanity-check command above works; if it does, the issue is likely a domain gap between your target image and the high-fidelity training data.
- **Batch sweep empty** – Inspect the CSV (if generated) to confirm foreground pixels are truly absent; otherwise revisit preprocessing (try `--clahe`) or consider fine-tuning on frames closer to the target domain.
- **CUDA missing** – Supply `--device cpu` if you run on hardware without CUDA drivers.
- **Palette differences** – Colours come from `label_colors.py`. If you modify that mapping, both training overlays and inference previews stay in sync automatically.
