# UNet Training Command

The `spine train-unet` command fits a UNet baseline on the pseudo lateral dataset and now provides the tooling required for serious experiments: train/val/test splits, on-the-fly augmentations, mixed precision, richer metrics, early stopping, and structured logging.

## Expected dataset layout

Run `scripts/build_pseudo_dataset.sh` to generate angle folders. Each angle should contain:

```
angles/<rot_id>/manifest.csv        # two columns: image, mask_labels
angles/<rot_id>/images/<stem>.png
angles/<rot_id>/mask_labels/<stem>.png
...
```

The manifest is the only input required by the training command.

## Data preparation

Prepare centred tiles first:

```
poetry run spine prepare-unet-tiles \
  --csv prepared/unet_tiles/c4/manifest.csv \
  --out-dir prepared/unet_tiles/c4
```

Use the generated manifest in the next step.

## Usage

```
poetry run spine train-unet \
  --csv prepared/unet_tiles/c4/manifest.csv \
  --out-dir runs/unet_exp01 \
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

### Key options

| Flag | Description |
|------|-------------|
| `--csv` | Path to a manifest CSV (must contain `image` and `mask_labels`). |
| `--out-dir` | Directory used for checkpoints, logs, and reports (created if missing). |
| `--epochs` | Maximum number of epochs to run (default: 5). |
| `--batch-size` | Mini-batch size (default: 4). |
| `--learning-rate` | Adam learning rate (default: 1e-3). |
| `--val-fraction` | Fraction of the dataset reserved for validation (default: 0.1, set to 0 to disable). |
| `--test-fraction` | Fraction reserved for a held-out test set that is only evaluated after training (default: 0.1). |
| `--num-workers` | Number of DataLoader workers (default: 2). |
| `--device` | Training device string (defaults to `cuda` if present, otherwise `cpu`). |
| `--seed` | Global RNG seed (Python, NumPy, PyTorch). |
| `--patience` | Early-stopping patience in epochs (default: 5). Set to 0 to disable. |
| `--min-delta` | Required improvement in the monitored metric before patience resets (default: 0). |
| `--monitor` | Metric used for checkpointing/early stopping: `dice`, `iou`, `f1`, or `loss` (default: `dice`). |
| `--clip-grad-norm` | Gradient-norm clipping value (default: 1.0, set ≤0 to disable). |
| `--amp/--no-amp` | Toggle CUDA mixed precision (enabled by default when CUDA is available). |
| `--augment/--no-augment` | Toggle training-time augmentations (enabled by default). |
| `--lr-factor`, `--lr-patience`, `--lr-min` | ReduceLROnPlateau scheduler controls (defaults: 0.5, 2, 1e-6). |
| `--log-dir` | Optional TensorBoard log directory (defaults to `<out-dir>/tensorboard`). |

## Training behaviour

- **Splits** – The manifest is shuffled with the configured seed, then split into train/validation/test partitions according to `--val-fraction` and `--test-fraction`. Validation is used for early stopping and scheduler updates; the test set is only evaluated once the best checkpoint is determined.
- **Augmentations** – When enabled, the training loader applies horizontal flips, small ±10° rotations (nearest-neighbour for masks), mild brightness/contrast shifts, and low-magnitude Gaussian noise. Validation/test remain untouched.
- **Mixed precision** – CUDA runs default to AMP for faster convergence; CPUs fall back to FP32 automatically. AMP can be disabled with `--no-amp`.
- **Optimiser & scheduler** – Adam + `ReduceLROnPlateau` (on validation loss). The scheduler parameters are exposed (`--lr-factor`, `--lr-patience`, `--lr-min`).
- **Gradient clipping** – The global norm is clipped to `--clip-grad-norm` after unscaling (useful when the loss spikes).
- **Metrics** – Each epoch reports loss, accuracy, macro precision/recall/F1, Dice, and IoU for train and validation splits. Per-class scores are stored in the history JSON. Test metrics are computed with the best checkpoint.
- **Checkpointing & early stopping** – The model with the best monitored metric is written to `unet_best.pt`. Patience counts epochs where the monitored metric fails to improve by at least `min_delta`.

## Outputs

Running `train-unet` populates the output directory with:

- `unet_best.pt` – PyTorch checkpoint containing the best-model weights, class count, and config snapshot.
- `metrics.csv` – Tabular log of epoch-level loss and macro metrics for train/val/test (ready for Pandas or spreadsheets).
- `training_summary.json` – Full record of hyper-parameters, metric history (including per-class metrics), best epoch, and optional test diagnostics.
- `tensorboard/` (when TensorBoard is installed) – Event files mirroring the CSV metrics for interactive plotting.

The command prints per-epoch summaries and announces when early stopping triggers.

## Interpreting the metrics

- **Macro vs per-class** – Macro scores average across labels that appear in ground truth, preventing large vertebrae from dominating the summary. Per-class precision/recall/F1/Dice/IoU live under `history[n]['train'/'val']['per_class']` in `training_summary.json`.
- **Dice vs IoU** – Dice tends to be smoother for thin structures; IoU is stricter. Monitor whichever captures your project goals best (`--monitor`).
- **Precision & Recall** – Useful to catch over/under-segmentation. Precision high + recall low means we miss vertebra pixels; the opposite indicates leakage into background.
- **Learning rate** – `metrics.csv` and the history entries include the effective LR each epoch so scheduler behaviour is traceable.

## Architecture & loss

- UNet encoder–decoder with channel widths `[32, 64, 128, 256, 512]` (same as before) producing multi-class logits.
- Images are single-channel, normalised to `[0, 1]`; masks remain integer-encoded labels.
- `CrossEntropyLoss` supervises the logits directly. Metrics are thresholded by argmax to form discrete masks.

## Tips

- Disable augmentations (`--no-augment`) for pure benchmarking or deterministic regression tests.
- Switch the monitor to `loss` if you prefer loss-driven stopping, or to `iou`/`f1` depending on reporting needs.
- Leave the test split untouched during development; evaluate once per experiment to avoid leakage.
- The JSON + CSV combo makes it easy to plot learning curves in notebooks, spreadsheet tools, or TensorBoard (if installed).
