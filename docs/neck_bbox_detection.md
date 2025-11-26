# Neck BBox Detection (Step 1 follow-up)

Goal: train a simple detector that, given a binarized VFSS-like frame, returns a neck bbox so we can crop/letterbox the frame before segmentation.

## 1) Export YOLO dataset from synthetic outputs
- Make sure you generated synthetic neck projections with `--save-bbox`; each run writes `bbox_labels.json` and the processed images (e.g., `deepdrr/otsu_overlay.png`).
- Export to YOLO format (single class `neck`) using the helper:
  ```
  python scripts/export_neck_bbox_yolo.py \
    --input-root outputs/dataset_synth_headneck \
    --output-root prepared/neck_bbox_yolo \
    --image-key otsu_overlay \
    --val-ratio 0.2
  ```
  `image-key` options:
  - `otsu_overlay` (default): binarized overlay after CLAHE2+blur+Otsu — matches what we’ll feed the detector at inference.
  - `clahe2`: contrast-only grayscale if you want to avoid binary inputs.
  - `deepdrr`: raw DRR (no Otsu).

## 2) Train a lightweight detector (YOLOv8 example)
- Install once: `pip install ultralytics` (or use any detector you prefer).
- Create `prepared/neck_bbox_yolo/data.yaml`:
  ```yaml
  path: prepared/neck_bbox_yolo
  train: train/images
  val: val/images
  nc: 1
  names: [neck]
  ```
- Train:
  ```
  yolo detect train model=yolov8n.pt data=prepared/neck_bbox_yolo/data.yaml \
    epochs=80 imgsz=384 batch=16 project=runs/neck_bbox
  ```
  Swap `yolov8n.pt` for a larger model if needed.

## 3) Run detector on reference frames and crop
- Preprocess frames the same way as training inputs (CLAHE2+blur+Otsu if you trained on `otsu_overlay`). You can reuse `deepdrr-pair`’s `_process_single` logic; simplest: call `python - <<'PY' ... >>` as below.
- Detect and save predictions (YOLOv8):
  ```
  yolo detect predict model=runs/neck_bbox/train/weights/best.pt \
    source=data/frames/50 \
    imgsz=384 conf=0.25 save_txt=True save_conf=True \
    project=runs/neck_bbox_infer
  ```
  This writes `labels/*.txt` with normalized bboxes alongside images in `runs/neck_bbox_infer/predict/`.
- Crop + letterbox using the detected bbox (single class) and save to a new directory:
  ```python
  import cv2, json
  from pathlib import Path

  pred_dir = Path("runs/neck_bbox_infer/predict")
  out_dir = Path("outputs/frames_cropped")
  out_dir.mkdir(parents=True, exist_ok=True)

  for img_path in (pred_dir / "images").glob("*.png"):
      stem = img_path.stem
      label_path = pred_dir / "labels" / f"{stem}.txt"
      if not label_path.exists():
          continue
      with open(label_path) as f:
          line = f.readline().strip()
      if not line:
          continue
      _, cx, cy, w, h = map(float, line.split())
      img = cv2.imread(str(img_path))
      H, W = img.shape[:2]
      x0 = int((cx - w/2) * W); y0 = int((cy - h/2) * H)
      x1 = int((cx + w/2) * W); y1 = int((cy + h/2) * H)
      x0 = max(0, x0); y0 = max(0, y0); x1 = min(W-1, x1); y1 = min(H-1, y1)
      crop = img[y0:y1+1, x0:x1+1]
      # Letterbox back to original size
      def letterbox(im, tw, th, color=0):
          h, w = im.shape[:2]
          scale = min(tw/w, th/h)
          nw, nh = int(w*scale), int(h*scale)
          resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
          pad_w, pad_h = tw - nw, th - nh
          top, bottom = pad_h//2, pad_h - pad_h//2
          left, right = pad_w//2, pad_w - pad_w//2
          return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
      boxed = letterbox(crop, W, H, color=0)
      cv2.imwrite(str(out_dir / f"{stem}_crop.png"), boxed)
  ```
  Adjust `color` if you want non-black padding.

## Notes and rationale
- Training on binarized overlays keeps the detector focused on neck silhouette and bone shape, matching the intended inference preproc.
- If your detector underfits, increase `epochs`, move to `yolov8s.pt`, or add simple augmentations (flip, rotate).
- To change the target crop size, swap the letterbox dimensions (e.g., fixed 384×384 instead of original frame size).***

## Curate cropped real frames into YOLO (CLI)
- We keep manual review fully in the terminal: the curator prints an ASCII preview (no OpenCV windows) and asks whether to keep each crop.
- Expected inputs: side-by-side panels in `outputs/frames_cropped_iter/side_by_side/*_side.png` and their matching crops in `outputs/frames_cropped_iter/crops/*_crop.png`.
- Order: files are iterated in natural numeric order (e.g., `f0, f1, f2, ... f10`), so frame sequences are reviewed chronologically.
- Run (adjust split/prefix as needed):
  ```bash
  python scripts/curate_cropped_frames_to_yolo.py \
    --side-dir outputs/frames_cropped_iter/side_by_side \
    --crop-dir outputs/frames_cropped_iter/crops \
    --dest-root prepared/neck_bbox_yolo \
    --split train \
    --prefix real_ \
    --preview-width 120
  ```
- Controls: single-key, no Enter. `a/add` accepts and writes `images/` (CLAHE2+Otsu mask) and `labels/` (YOLO bbox); `s/skip` ignores; `q` quits; `h/?` shows help. Set `--preview-width 0` to disable ASCII if you only want prompts.
- The script recomputes CLAHE2 + Gaussian blur + Otsu on the crop before saving, so the saved image matches the detector input we train on.

## Option A: Train on full-frame masks (no post-crop reprocessing)
- Preprocess raw frames once with the training pipeline (CLAHE2 + blur + Otsu), keeping full resolution:
  ```bash
  poetry run python scripts/preprocess_frames_for_yolo.py \
    --input-dir data/frames/50 \
    --out-dir prepared/frames50_otsu \
    --output-kind mask \
    --clip-limit1 2.0 --clip-limit2 2.0 \
    --tile-size1 8 --tile-size2 8 \
    --blur-kernel 5
  ```
- Export to YOLO using the full masks (single class `neck`), deriving bboxes from foreground pixels:
  ```bash
  poetry run python scripts/export_full_masks_to_yolo.py \
    --mask-dir prepared/frames50_otsu \
    --output-root prepared/neck_bbox_yolo_full \
    --val-ratio 0.2 \
    --prefix realfull_
  ```
- Train on `prepared/neck_bbox_yolo_full/data.yaml` (create YAML pointing to `train/images` and `val/images`):
  ```yaml
  path: prepared/neck_bbox_yolo_full
  train: train/images
  val: val/images
  nc: 1
  names: [neck]
  ```
  Then run:
  ```bash
  poetry run yolo detect train \
    model=yolov8n.pt \
    data=prepared/neck_bbox_yolo_full/data.yaml \
    imgsz=384 batch=16 epochs=80 \
    project=runs/neck_bbox name=yolov8n_fullmask
  ```
- Inference: preprocess new frames the same way (`preprocess_frames_for_yolo.py` to masks), then `yolo detect predict` on those full masks. This keeps train/infer preprocessing identical and avoids the crop-vs-full mismatch.
