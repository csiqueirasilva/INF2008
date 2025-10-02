#!/usr/bin/env bash
set -euo pipefail

# Build pseudo‑lateral dataset for a single subset (HNSCC-3DCT-RT),
# generating multiple angles via small yaw/pitch/roll sweeps.
# Also increases slab sampling to 30 within the same ±keep-mm window.

# Locate CLI entry point (allows overriding via SPINE_BIN env).
if [[ -n "${SPINE_BIN:-}" ]]; then
  # shellcheck disable=SC2206
  SPINE_CMD=(${SPINE_BIN})
elif command -v spine >/dev/null 2>&1; then
  SPINE_CMD=(spine)
elif command -v poetry >/dev/null 2>&1; then
  SPINE_CMD=(poetry run spine)
else
  echo "Cannot find 'spine' CLI. Install it or set SPINE_BIN='poetry run spine'" >&2
  exit 1
fi

# Fixed config
SUBSET="HNSCC-3DCT-RT"
LABELS="1-7"
PLANE="sag"

# Base orientation (deg). We will sweep small deltas around these.
BASE_YAW=0
BASE_PITCH=90
BASE_ROLL=0
ROT90=${ROT90:-0}
FLIP_H=${FLIP_H:-0}
FLIP_V=${FLIP_V:-0}
APERTURE=0
AXIS_HELPER=${AXIS_HELPER:-0}

# Tone / projection defaults (mimic inspect 00_* CT look)
CT_WINDOW_LO=${CT_WINDOW_LO:--1000}
CT_WINDOW_HI=${CT_WINDOW_HI:-1000}
PROJECTION_POWER=${PROJECTION_POWER:-1.0}
TONE_STYLE=${TONE_STYLE:-ct}

# Cropping around subject (disabled by default while tuning orientation)
AUTO_CROP=${AUTO_CROP:-0}
CROP_MARGIN_MM=${CROP_MARGIN_MM:-20}
RESIZE_AFTER_CROP=${RESIZE_AFTER_CROP:-0}

# Pan (mm)
PAN_X=0
PAN_Y=0
PAN_Z=0

# Depth sampling
SLAB_MM=${SLAB_MM:-12}
KEEP_MM=${KEEP_MM:-0} # was 12
WINDOW_COUNT=${WINDOW_COUNT:-1} # was 30; 
SLAB_COUNT=${SLAB_COUNT:-1}

# Angle sweeps (deg) as min:max:step around base angles
PITCH_SWEEP=${PITCH_SWEEP:--3:3:1}
YAW_SWEEP=${YAW_SWEEP:-0:0:1}
ROLL_SWEEP=${ROLL_SWEEP:-0:0:1}

# Paths and misc
DATA_ROOT=${DATA_ROOT:-data/CTSpine1K}
OUT_DIR=${OUT_DIR:-data/pseudo_lateral}
LIMIT_CASES=${LIMIT_CASES:-0}     # 0 = all
OVERRIDE_EXISTING=${OVERRIDE_EXISTING:-0}
CLEAR_DIR=${CLEAR_DIR:-0}

usage() {
  cat <<EOF
Usage: $0 [--data-root DIR] [--out-dir DIR] [--limit-cases N] [--override-existing] [--clear-dir] [--axis-helper]
          [--pitch-sweep a:b:s[,a:b:s...]] [--yaw-sweep a:b:s[,a:b:s...]] [--roll-sweep a:b:s[,a:b:s...]]

Generates only subset '${SUBSET}' with fixed parameters and angle sweeps around base yaw=0, pitch=90, roll=-90.
Defaults: pitch sweep -3..+3 by 1 deg; yaw and roll sweeps disabled (0..0).
You can pass multiple ranges separated by commas, e.g. --pitch-sweep "87:93:1,267:273:1".
Environment variables DATA_ROOT/OUT_DIR/LIMIT_CASES/OVERRIDE_EXISTING/CLEAR_DIR/AXIS_HELPER/PITCH_SWEEP/YAW_SWEEP/ROLL_SWEEP can set defaults.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT=$2; shift 2;;
    --out-dir) OUT_DIR=$2; shift 2;;
    --limit-cases) LIMIT_CASES=$2; shift 2;;
    --override-existing) OVERRIDE_EXISTING=1; shift;;
    --clear-dir) CLEAR_DIR=1; shift;;
    --axis-helper) AXIS_HELPER=1; shift;;
    --pitch-sweep) PITCH_SWEEP=$2; shift 2;;
    --yaw-sweep) YAW_SWEEP=$2; shift 2;;
    --roll-sweep) ROLL_SWEEP=$2; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# If asked, clear the ROOT out dir once before generating any angles
if [[ "$CLEAR_DIR" -eq 1 ]]; then
  if [[ -n "$OUT_DIR" && "$OUT_DIR" != "/" && -d "$OUT_DIR" ]]; then
    # basic safety: require at least 3 path components or a recognizable prefix
    case "$OUT_DIR" in
      /*/*/*|data/*|outputs/*)
        echo "🧹 Clearing root out dir: $OUT_DIR"
        rm -rf -- "$OUT_DIR"
        ;;
      *)
        echo "Refusing to clear suspicious OUT_DIR path: '$OUT_DIR'" >&2
        exit 1
        ;;
    esac
  fi
fi

mkdir -p "$OUT_DIR"

# Expand sweep specs.
# Accepts comma-separated list of ranges (a:b:s) or single numbers.
# Prints one value per line; preserves order; removes duplicates.
expand_sweep() {
  local spec="$1"
  local IFS=','
  read -r -a parts <<< "$spec"
  for part in "${parts[@]}"; do
    part=$(echo "$part" | xargs)
    [[ -z "$part" ]] && continue
    if [[ "$part" =~ ^-?[0-9]+:-?[0-9]+:-?[0-9]+$ ]]; then
      local a b s
      IFS=':' read -r a b s <<< "$part"
      [[ -z "$s" || "$s" == 0 ]] && s=1
      if (( s < 0 )); then s=$((-s)); fi
      if (( a <= b )); then
        seq "$a" "$s" "$b"
      else
        seq "$a" "-$s" "$b"
      fi
    elif [[ "$part" =~ ^-?[0-9]+$ ]]; then
      echo "$part"
    fi
  done | awk 'BEGIN{FS="\n"} !seen[$0]++'
}

mapfile -t P_SEQ < <(expand_sweep "$PITCH_SWEEP")
mapfile -t Y_SEQ < <(expand_sweep "$YAW_SWEEP")
mapfile -t R_SEQ < <(expand_sweep "$ROLL_SWEEP")

echo "==> Subset=$SUBSET labels=$LABELS slabs=$WINDOW_COUNT within ±${KEEP_MM}mm"
echo "==> Sweeps: pitch {${P_SEQ[*]}}; yaw {${Y_SEQ[*]}}; roll {${R_SEQ[*]}} (base yaw=${BASE_YAW}, pitch=${BASE_PITCH}, roll=${BASE_ROLL})"

for dp in "${P_SEQ[@]}"; do
  for dy in "${Y_SEQ[@]}"; do
    for dr in "${R_SEQ[@]}"; do
      YAW=$((BASE_YAW + dy))
      PITCH=$((BASE_PITCH + dp))
      ROLL=$((BASE_ROLL + dr))
      OUT_SUBDIR="$OUT_DIR/angles/p$(printf '%+d' "$PITCH")_y$(printf '%+d' "$YAW")_r$(printf '%+d' "$ROLL")"
      mkdir -p "$OUT_SUBDIR"
      echo "--> Angle yaw=$YAW pitch=$PITCH roll=$ROLL → out=$OUT_SUBDIR"

      "${SPINE_CMD[@]}" build-pseudo-lateral \
        --data-root "$DATA_ROOT" \
        --subset "$SUBSET" \
        --limit-cases "$LIMIT_CASES" \
        --labels "$LABELS" \
        --plane "$PLANE" \
        --yaw "$YAW" --pitch "$PITCH" --roll "$ROLL" \
        --rot90 "$ROT90" \
        --ct-window-lo "$CT_WINDOW_LO" \
        --ct-window-hi "$CT_WINDOW_HI" \
        --projection-power "$PROJECTION_POWER" \
        --tone-style "$TONE_STYLE" \
        $([ "$AUTO_CROP" -eq 1 ] && echo "--auto-crop" || echo "--no-auto-crop") \
        --crop-margin-mm "$CROP_MARGIN_MM" \
        $([ "$RESIZE_AFTER_CROP" -eq 1 ] && echo "--resize-after-crop" || echo "--no-resize-after-crop") \
        $([ "$FLIP_H" -eq 1 ] && echo "--flip-h" || true) \
        $([ "$FLIP_V" -eq 1 ] && echo "--flip-v" || true) \
        $([ "$APERTURE" -eq 1 ] && echo "--aperture" || echo "--no-aperture") \
        $([ "$AXIS_HELPER" -eq 1 ] && echo "--axis-helper" || true) \
        $([ "$CLEAR_DIR" -eq 1 ] && echo "--clear-dir" || true) \
        --pan-x-mm "$PAN_X" --pan-y-mm "$PAN_Y" --pan-z-mm "$PAN_Z" \
        --keep-mm "$KEEP_MM" --window-count "$WINDOW_COUNT" \
        --slab-mm "$SLAB_MM" \
        --slab-count "$SLAB_COUNT" \
        $([ "$OVERRIDE_EXISTING" -eq 1 ] && echo "--override-existing" || true) \
        --out-dir "$OUT_SUBDIR"

    done
  done
done

echo "All done. Manifests under $OUT_DIR/angles/*/manifest.csv"
