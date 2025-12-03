# Inference Speed Benchmark

Use `scripts/benchmark_inference_speed.py` to time the full neck pipeline (CLAHE2 frame → neck YOLO → crop/letterbox → UNet). It loads the model config from the checkpoint, so multi-channel inputs (CLAHE2 + Otsu + coords) are honored automatically.

## Run
Example (CUDA, vid7 clahe crop):
```bash
poetry run python scripts/benchmark_inference_speed.py \
  --frame prepared/frames7_clahe2/v7_f129_clahe2.png \
  --neck-yolo runs/neck_bbox/yolov8n_circ62/weights/best.pt \
  --unet runs/unet_cropped6_clahe_pos/unet_best.pt \
  --device cuda \
  --runs 30 --warmup 5 \
  --imgsz 384 --crop-size 1024 --conf 0.25 --iou 0.6
```
Output: mean/median/P90 total latency per frame (ms) across runs. YOLO and UNet forward passes are on the chosen device; preprocessing (Otsu/coords) runs on CPU inside the timed block.

## Notes
- Uses the checkpoint’s stored channel config (`use_otsu_channel`, `use_coord_channels`, `otsu_blur_kernel`, `num_input_channels`) so the input tensor matches training.
- Letterbox size defaults to 1024; adjust `--crop-size` if your UNet was trained at a different resolution.
- Neck YOLO is run with `max_det=1`; if no box is found, the full frame is used.
- Timings include crop/letterbox + UNet tensor prep; disk I/O for the frame is outside the timed loop.***
