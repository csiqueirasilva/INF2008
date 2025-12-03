# Patient-Level Splits, Poison Prevention, and GT Evaluation

Goal: ensure splits are at the patient level, prevent leakage/poisoning, and compute IoU/Dice on held-out real patients with ground truth masks.

## 1) Patient ID and Manifests
- Derive `patient_id` from paths (e.g., first component matching `HN_P...` in the DeepDRR path).
- Include `patient_id` in all manifests (`image,mask_labels,patient_id` for UNet; `image,patient_id` for YOLO exports).

## 2) Patient-Level Split
```python
import random

# patients
patients = sorted({row["patient_id"] for row in samples})
rng = random.Random(seed)
rng.shuffle(patients)
n = len(patients)
n_train = int(0.75 * n)
n_val   = int(0.15 * n)
train_ids = set(patients[:n_train])
val_ids   = set(patients[n_train:n_train+n_val])
test_ids  = set(patients[n_train+n_val:])
assert train_ids.isdisjoint(val_ids | test_ids)

train = [s for s in samples if s["patient_id"] in train_ids]
val   = [s for s in samples if s["patient_id"] in val_ids]
test  = [s for s in samples if s["patient_id"] in test_ids]
```
- Enforce this in all exporters/trainers (UNet, YOLO).
- Log the patient partitions (train/val/test) for reproducibility and auditing.

## 3) Poison Prevention
- Refuse to proceed if a `patient_id` appears in multiple splits.
- Fix `seed` and record the patient lists used per run.
- Publish the patient split lists alongside results.

## 4) Holdout Patient GT Evaluation
To get leakage-free IoU/Dice on real frames with GT:
1. Pick a holdout patient (or rotate through several). Exclude all their data from train/val/test splits during training.
2. Train models (neck YOLO + UNet) on the remaining patients.
3. Run the full pipeline on the holdout’s frames (preprocess → neck YOLO → crop → UNet).
4. Compute IoU/Dice against GT masks for that holdout only.

Command sketch (per holdout):
```bash
# After training without patient P_holdout
poetry run spine predict-unet-batch \
  --model runs/unet_cropped6_clahe_pos/unet_best.pt \
  --image-dir outputs/frames<P_holdout>_crops_clahe/crops \
  --pattern "*_crop.png" \
  --overlay-dir outputs/frames<P_holdout>_crops_clahe/unet_overlays_pos \
  --mask-dir outputs/frames<P_holdout>_crops_clahe/unet_masks_pos \
  --no-clahe \
  --use-otsu-channel --use-coord-channels --otsu-blur-kernel 5

poetry run python scripts/eval_unet_iou.py \
  --gt-dir gt_masks/<P_holdout> \
  --pred-dir outputs/frames<P_holdout>_crops_clahe/unet_masks_pos \
  --pred-suffix "_mask" \
  --classes 1,2,4,8,16,32,64
```

## 5) Cross-Patient Rotation
- Repeat the holdout for each patient with available GT (leave-one-patient-out or K-fold by patient).
- Aggregate mean/median IoU/Dice and ranges across folds.
- Record patient splits for every fold to demonstrate leakage-free evaluation.

## 6) Reporting
- State that splits are by patient ID; no patient appears in more than one split.
- For GT evaluation, report per-patient and aggregate IoU/Dice from the holdout runs.
- Provide the patient split lists and seeds used.***
