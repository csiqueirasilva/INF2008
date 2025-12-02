# Full-Frame → YOLO → Cropped UNet Pipeline

Goal: detect C1–C7 on VFSS-like frames. Train YOLO on full-frame circular synthetics to crop the neck, then train a UNet on the cropped panels to predict a multi-class bitmask (C1–C7) on the crop.

## 1) Prepare labeled CT data
- Source: CTSpine1k with vertebra labels (C1–C7). DeepDRR consumes the CT volume + label map to render projections and label overlays.

## 2) Render synthetic VFSS with DeepDRR (circular + cropped views)
- Use `deepdrr_pair` (e.g., via `build_pseudo_dataset_step1of2.sh`) with:
  - Aperture enabled to get a circular view (`deepdrr/circular-synth.png`).
  - Bbox crop enabled so the final outputs are neck-centered/letterboxed.
  - Saves bboxes (`bbox_labels.json`) and label bitmasks:
    - `deepdrr/label_bitmask_circular_synth.png` – aligned to `circular-synth.png` (pre-crop aperture view).
    - `deepdrr/label_bitmask_cropped_letterboxed.png` – aligned to the cropped/letterboxed outputs (e.g., `deepdrr/otsu_overlay.png`).
- Final images per run (after zoom/aperture/crop/letterbox):
  - Full circular (before Otsu): `deepdrr/circular-synth.png`
  - Cropped DRR variants: `deepdrr/clahe2.png`, `deepdrr/otsu_overlay.png`, etc.

## 3) Image preparation (for detector & UNet inputs)
- Apply CLAHE twice (`clipLimit=2.0`, `tileGridSize=8×8`), Gaussian blur (`k=5`), then Otsu binarization. This matches:
  - YOLO training inputs when exporting with `--image-key circular_synth_otsu`.
  - UNet training inputs when using the cropped Otsu outputs.

## 4) Build training datasets
### 4a) YOLO (full-frame circular masks)
Export circular-synth with CLAHE2+blur+Otsu to YOLO format:
```bash
poetry run python scripts/export_neck_bbox_yolo.py \
  --input-root outputs/dataset_synth_headneck_2 \
  --output-root prepared/neck_bbox_yolo_circular \
  --image-key circular_synth_otsu \
  --val-ratio 0.2 \
  --clip-limit1 2.0 --clip-limit2 2.0 \
  --tile-size1 8 --tile-size2 8 \
  --blur-kernel 5
```
This writes full-frame circular Otsu images + YOLO bboxes from `bbox_labels.json`.

### 4b) UNet (cropped panels → bitmask)
Use the cropped Otsu images and the cropped bitmask as pairs. Build a manifest (`image,mask_labels`) pointing to:
- Image: `.../deepdrr/otsu_overlay.png` (cropped/letterboxed Otsu view).
- Mask: `.../deepdrr/label_bitmask_cropped_letterboxed.png` (uint16 bitmask; bits map to labels in order).

Example manifest builder:
```bash
poetry run python - <<'PY'
from pathlib import Path
import csv
runs = Path("outputs/dataset_synth_headneck_2")
pairs = []
for bm in runs.rglob("deepdrr/label_bitmask_cropped_letterboxed.png"):
    img = bm.parent / "otsu_overlay.png"
    if img.exists():
        pairs.append((img, bm))
out = Path("prepared/unet_cropped/manifest.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "mask_labels"])
    for img, mask in pairs:
        w.writerow([img.as_posix(), mask.as_posix()])
print(f"wrote {len(pairs)} pairs to {out}")
PY
```

## 5) Train models
### YOLO (cropper)
```bash
poetry run yolo detect train \
  model=yolov8n.pt \
  data=prepared/neck_bbox_yolo_circular/data.yaml \
  imgsz=384 batch=16 epochs=80 \
  project=runs/neck_bbox name=yolov8n_circular
```

### UNet (segmenter on crops)
```bash
poetry run spine train-unet \
  --csv prepared/unet_cropped/manifest.csv \
  --out-dir runs/unet_cropped \
  --epochs 40 --batch-size 6 \
  --learning-rate 5e-4 \
  --val-fraction 0.15 --test-fraction 0.1 \
  --monitor dice --patience 8 --amp --augment
```
Inputs are single-channel Otsu crops; targets are the bitmask labels.

## 6) Inference on real VFSS
1. Preprocess frames (CLAHE2 → blur → Otsu) to get masks matching training:
```bash
poetry run python scripts/preprocess_frames_for_yolo.py \
  --input-dir data/frames/<id> \
  --out-dir prepared/frames<id>_otsu \
  --output-kind mask
```
2. Detect neck bbox on full masks (max 1 box):
```bash
poetry run yolo detect predict \
  model=runs/neck_bbox/yolov8n_circular/weights/best.pt \
  source="prepared/frames<id>_otsu/*_mask.png" \
  imgsz=384 conf=0.25 max_det=1 save_txt save_conf save \
  project=runs/neck_bbox_infer name=frames<id>_circular
```
3. Crop the original frame using YOLO labels (letterbox to model size); reuse the same CLAHE2→Otsu on the crop if needed to mirror training, or feed the crop Otsu produced by the preprocessing step if you save it alongside.
4. Run the UNet on the cropped Otsu image to get the bitmask prediction (argmax over channels).

## 7) Output
- YOLO: bbox of neck on full-frame mask.
- UNet: per-pixel vertebra bitmask on the cropped panel (C1–C7), which can be recoloured or overlaid for visualization.

## Debugging IoU on training outputs
- To verify the UNet is learning correctly, run inference on the training crops and compute IoU/Dice against the GT bitmasks:
  ```bash
  poetry run spine predict-unet-batch \
    --model runs/unet_cropped_flat/unet_best.pt \
    --image-dir prepared/unet_cropped_flat/images \
    --overlay-dir outputs/unet_cropped_flat_train_overlays \
    --mask-dir outputs/unet_cropped_flat_train_preds \
    --pattern "*.png"

  poetry run python scripts/eval_unet_iou.py \
    --gt-dir prepared/unet_cropped_flat/masks \
    --pred-dir outputs/unet_cropped_flat_train_preds \
    --classes 1,2,4,8,16,32,64
  ```
- This reports per-class IoU (C1–C7) and mean IoU over the training set to confirm alignment between inputs and labels.

### Visual GT vs Pred overlays
- To create YOLO-style side-by-side overlays (GT on the left, pred on the right) for inspection:
  ```bash
  poetry run python scripts/overlay_gt_pred_comparison.py \
    --gt-dir prepared/unet_cropped_flat/masks \
    --pred-dir outputs/unet_cropped_flat_train_preds \
    --image-dir prepared/unet_cropped_flat/images \
    --pred-suffix _mask \
    --out-dir outputs/unet_cropped_flat_gt_pred
  ```
- Overlays are saved as `*_gt_pred.png`, useful for presentations and qualitative checks.
