#!/usr/bin/env bash
set -euo pipefail

# Generate DeepDRR slices for a patient, optionally clean masks, rebuild crops, and write bbox overlays.
# Default behavior: generate, clean in a copy, rebuild crops, overlays on clahe2.
#
# Example:
# bash scripts/build_pseudo_dataset_all_slices.sh \
#   --patient-id HN_P007 \
#   --out-root outputs/dataset_synth_headneck_6_full \
#   --frame data/frames/50/v50_f145.png \
#   --stride 1 --offset-min -1 --offset-max 1 \
#   --clean --clean-inplace --clean-kernel 7 --clean-iters 1 --clean-min-area 200

CT_ROOT=data/CTSpine1K/raw_data/volumes/HNSCC-3DCT-RT
LBL_ROOT=data/CTSpine1K/raw_data/labels/HNSCC-3DCT-RT
FRAME=data/frames/50/v50_f145.png
OUT_ROOT=outputs/dataset_synth_headneck_all
PIXEL_MM=0.7
SLICE_THICKNESS_MM=0   # 0 => full volume; >0 => thin slab thickness
STRIDE=1
PATIENT_ID=""
OFFSET_MIN=""
OFFSET_MAX=""
CLEAN=0
CLEAN_KERNEL=7
CLEAN_ITERS=1
CLEAN_MIN_AREA=0
CLEAN_INPLACE=0
REBUILD_CROPS=1
BBOX_OVERLAYS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --patient-id) PATIENT_ID="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --frame) FRAME="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --offset-min) OFFSET_MIN="$2"; shift 2 ;;
    --offset-max) OFFSET_MAX="$2"; shift 2 ;;
    --clean) CLEAN=1; shift 1 ;;
    --clean-kernel) CLEAN_KERNEL="$2"; shift 2 ;;
    --clean-iters) CLEAN_ITERS="$2"; shift 2 ;;
    --clean-min-area) CLEAN_MIN_AREA="$2"; shift 2 ;;
    --clean-inplace) CLEAN_INPLACE=1; shift 1 ;;
    --no-rebuild-crops) REBUILD_CROPS=0; shift 1 ;;
    --no-bbox-overlays) BBOX_OVERLAYS=0; shift 1 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$PATIENT_ID" ]]; then
  echo "Please set --patient-id (e.g., HN_P001)"; exit 1
fi

CT_PATH="$CT_ROOT/${PATIENT_ID}.nii.gz"
LBL_PATH="$LBL_ROOT/${PATIENT_ID}_seg.nii.gz"
if [[ ! -f "$CT_PATH" ]]; then
  echo "CT not found: $CT_PATH"; exit 1
fi
if [[ ! -f "$LBL_PATH" ]]; then
  echo "Label not found: $LBL_PATH"; exit 1
fi

echo "Computing slice offsets for $PATIENT_ID with stride $STRIDE ..."
OFFSETS=$(python - <<PY
import nibabel as nib
ct = nib.load("$CT_PATH")
spacing = ct.header.get_zooms()
nz = ct.shape[2]
stride = int("$STRIDE")
zs = list(range(0, nz, stride))
mid = (nz - 1) / 2.0
offsets = [(z - mid) * spacing[2] for z in zs]
off_min = "$OFFSET_MIN"
off_max = "$OFFSET_MAX"
if off_min:
    off_min = float(off_min)
    offsets = [o for o in offsets if o >= off_min]
if off_max:
    off_max = float(off_max)
    offsets = [o for o in offsets if o <= off_max]
print(" ".join(f"{o:.3f}" for o in offsets))
PY
)

mkdir -p "$OUT_ROOT"

for offset in $OFFSETS; do
  offset_clean=${offset#-}
  offset_clean=${offset_clean//./p}
  offset_clean=${offset_clean//-}
  if [[ "$offset" == -* ]]; then
    offset_tag="m${offset_clean}"
  else
    offset_tag="p${offset_clean}"
  fi

  poetry run spine deepdrr-pair \
    --ct "$CT_PATH" --label-ct "$LBL_PATH" --frame "$FRAME" \
    --yaw 180 --pitch 0 --roll 180 \
    --size 384 --sensor-width 384 --pixel-mm "$PIXEL_MM" --sdd 1400 \
    --no-native-resolution \
    --guide-labels 1-7 --no-guides \
    --crop-square \
    --aperture-radius-scale 0.5 --aperture-softness 0.0 --aperture-blur 0 \
    --aperture-inside 1.0 --aperture-outside 0.0 \
    --zoom-factor 2.25 --pan-x-px -25 --pan-y-px 30 \
    --no-noise --spectrum 60KV_AL35 --bone-scale 1.35 \
    --no-hist-match --no-edge-enhance \
    --otsu-source clahe2 --blur-kernel 5 \
    --slice-offset-mm "$offset" --slice-thickness-mm "$SLICE_THICKNESS_MM" \
    --save-bbox --crop-to-bbox --match-frame-size --crop-margin 0.05 \
    --letterbox-after-crop --letterbox-color 0 --letterbox-pad-only \
    --no-crop-frame \
    --colorize \
    --out-dir "$OUT_ROOT/$PATIENT_ID/off_${offset_tag}"
done

# Cleaning
TARGET_ROOT="$OUT_ROOT"
if [[ "$CLEAN" -eq 1 ]]; then
  echo "Cleaning masks with opening k=${CLEAN_KERNEL}, iters=${CLEAN_ITERS}, min-area=${CLEAN_MIN_AREA} ..."
  if [[ "$CLEAN_INPLACE" -eq 1 ]]; then
    poetry run python scripts/clean_bitmasks_priority.py \
      --root "$OUT_ROOT/$PATIENT_ID" \
      --kernel "$CLEAN_KERNEL" \
      --iters "$CLEAN_ITERS" \
      --min-component-area "$CLEAN_MIN_AREA" \
      --classes C1,C2,C3,C4,C5,C6,C7
    TARGET_ROOT="$OUT_ROOT"
  else
    TARGET_ROOT="${OUT_ROOT}_clean"
    poetry run python scripts/clean_bitmasks_priority.py \
      --root "$OUT_ROOT/$PATIENT_ID" \
      --out-root "$TARGET_ROOT/$PATIENT_ID" \
      --kernel "$CLEAN_KERNEL" \
      --iters "$CLEAN_ITERS" \
      --min-component-area "$CLEAN_MIN_AREA" \
      --classes C1,C2,C3,C4,C5,C6,C7
  fi
  echo "Cleaned output at $TARGET_ROOT/$PATIENT_ID"
fi

# Rebuild crops with cleaned masks
if [[ "$REBUILD_CROPS" -eq 1 ]]; then
  echo "Rebuilding crops to reflect cleaned bboxes ..."
  poetry run python scripts/rebuild_neck_crops.py \
    --input-root "$TARGET_ROOT/$PATIENT_ID" \
    --output-root "$TARGET_ROOT/$PATIENT_ID" \
    --image-name clahe2.png \
    --mask-name label_bitmask_cropped_letterboxed.png \
    --out-image-name clahe2.png \
    --out-mask-name label_bitmask_cropped_letterboxed.png \
    --target-size 1024 \
    --pad-frac 0.05
fi

# Overlays
if [[ "$BBOX_OVERLAYS" -eq 1 ]]; then
  echo "Generating bbox overlays in $TARGET_ROOT/$PATIENT_ID ..."
  poetry run python - "$TARGET_ROOT/$PATIENT_ID" <<'PY'
import cv2, numpy as np, sys
from pathlib import Path
root = Path(sys.argv[1])
colors={1:(0,255,0),2:(0,200,255),4:(255,128,0),8:(255,0,0),16:(128,0,255),32:(255,0,255),64:(255,255,0)}
count=0
for mask_path in root.rglob("label_bitmask_cropped_letterboxed.png"):
    img_path = mask_path.parent / "clahe2.png"
    if not img_path.exists():
        img_path = mask_path.parent / "circular-synth.png"
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if img is None or mask is None:
        continue
    ys,xs = np.nonzero(mask)
    if xs.size and ys.size:
        x0,y0,x1,y1 = xs.min(), ys.min(), xs.max(), ys.max()
        cv2.rectangle(img,(x0,y0),(x1,y1),(0,0,255),2)
        cv2.putText(img,"neck",(x0,max(0,y0-5)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2,cv2.LINE_AA)
    for bit,color in colors.items():
        if (mask & bit).max()==0:
            continue
        ys,xs=(mask & bit).nonzero()
        if not xs.size or not ys.size:
            continue
        x0,y0,x1,y1=xs.min(), ys.min(), xs.max(), ys.max()
        cv2.rectangle(img,(x0,y0),(x1,y1),color,2)
        cv2.putText(img,f"C{bit.bit_length()}",(x0,max(0,y0-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)
    cv2.imwrite(str(mask_path.parent/"bbox_overlay_clahe2.png"), img)
    count += 1
print(f"Wrote {count} overlays.")
PY
fi

echo "Done. Wrote to $TARGET_ROOT/$PATIENT_ID"
