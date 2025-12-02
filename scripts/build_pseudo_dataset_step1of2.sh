CT_ROOT=data/CTSpine1K/raw_data/volumes/HNSCC-3DCT-RT
LBL_ROOT=data/CTSpine1K/raw_data/labels/HNSCC-3DCT-RT
FRAME=data/frames/50/v50_f145.png
OUT_ROOT=outputs/dataset_synth_headneck_4
PIXEL_MM=0.1
# Set to 0 to project the full volume; >0 enables thin-slab mode (mm).
SLICE_THICKNESS_MM=0.0
OFFSETS_MM=("0.0")

mkdir -p "$OUT_ROOT"

for ct in "$CT_ROOT"/*.nii.gz; do
  base=$(basename "$ct" .nii.gz)
  lbl="$LBL_ROOT/${base}_seg.nii.gz"
  [ -f "$lbl" ] || { echo "Label not found for $base, skipping"; continue; }

  for offset in "${OFFSETS_MM[@]}"; do
    offset_clean=${offset#-}
    offset_clean=${offset_clean//./p}
    if [[ "$offset" == -* ]]; then
      offset_tag="m${offset_clean}"
    else
      offset_tag="p${offset_clean}"
    fi

    poetry run spine deepdrr-pair \
      --ct "$ct" --label-ct "$lbl" --frame "$FRAME" \
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
      --otsu-source clahe2 --blur-kernel 1 \
      --slice-offset-mm "$offset" --slice-thickness-mm "$SLICE_THICKNESS_MM" \
      --save-bbox --crop-to-bbox --match-frame-size --crop-margin 0.05 \
      --letterbox-after-crop --letterbox-color 0 --letterbox-pad-only \
      --no-crop-frame \
      --colorize \
      --out-dir "$OUT_ROOT/$base/off_${offset_tag}"
  done

done
