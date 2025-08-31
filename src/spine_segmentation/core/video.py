# src/spine_segmentation/core/video.py
from __future__ import annotations
import os, math
from pathlib import Path
from typing import List, Optional, Dict
import cv2
import numpy as np

try:
    import imageio.v2 as imageio  # fallback if OpenCV fails to open
except Exception:
    imageio = None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_stem(p: Path) -> str:
    # remove spaces etc. (helps consistent naming)
    return p.stem.replace(" ", "_")

def extract_frames_from_folder(
    input_dir: Path,
    output_dir: Path,
    img_ext: str = "png",
    every: int = 1,
    start: int = 0,
    max_per_video: int = 0,
    resize: Optional[int] = None,   # if set: square resize to e.g., 384
    gray: bool = False,
    keep_video_subdir: bool = False,
) -> List[Dict]:
    """
    Extract frames from all .avi videos in input_dir.

    Returns a manifest list with: {video, frame_idx, out_path, w, h, t_ms}
    """
    input_dir = Path(input_dir); output_dir = Path(output_dir)
    ensure_dir(output_dir)

    videos = sorted([p for p in input_dir.glob("*.avi")])
    if not videos:
        raise FileNotFoundError(f"No .avi in {input_dir}")

    manifest = []

    for vpath in videos:
        vname = safe_stem(vpath)
        subdir = output_dir / vname if keep_video_subdir else output_dir
        ensure_dir(subdir)

        # Try OpenCV first
        cap = cv2.VideoCapture(str(vpath))
        use_fallback = False
        if not cap.isOpened():
            use_fallback = True
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        written = 0
        idx = 0

        def write_frame(arr: np.ndarray, frame_idx: int, t_ms: float):
            nonlocal written
            if gray and len(arr.shape) == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            if resize:
                arr = cv2.resize(arr, (resize, resize), interpolation=cv2.INTER_AREA)
            # zero-padded index helps deterministic ordering
            fname = f"v{vname}_f{frame_idx}.{img_ext}"
            outp = subdir / fname
            # cv2.imwrite needs BGR or single channel
            ok = cv2.imwrite(str(outp), arr)
            if ok:
                h, w = (arr.shape[:2] if arr.ndim == 2 else arr.shape[:2])
                manifest.append(dict(
                    video=str(vpath), frame_idx=frame_idx, out_path=str(outp),
                    width=int(w), height=int(h), t_ms=float(t_ms)
                ))
                written += 1

        if not use_fallback:
            # OpenCV path
            # fast-skip frames by setting index
            for frame_idx in range(start, total):
                if (frame_idx - start) % max(1, every) != 0:
                    continue
                if max_per_video and written >= max_per_video:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                t_ms = (frame_idx / fps * 1000.0) if fps > 0 else (frame_idx * 1000.0 / 30.0)
                write_frame(frame, frame_idx, t_ms)
            cap.release()
        else:
            # Fallback via imageio if available
            if imageio is None:
                raise RuntimeError(f"Cannot open {vpath} with OpenCV; install imageio for fallback.")
            reader = imageio.get_reader(str(vpath))
            fps = reader.get_meta_data().get("fps", 30.0)
            for frame_idx, frame in enumerate(reader):
                if frame_idx < start or ((frame_idx - start) % max(1, every)) != 0:
                    continue
                if max_per_video and written >= max_per_video:
                    break
                # imageio returns RGB uint8; convert to BGR for cv2.imwrite consistency
                frame_bgr = frame[:, :, ::-1].copy()
                t_ms = frame_idx / fps * 1000.0
                write_frame(frame_bgr, frame_idx, t_ms)
            reader.close()

    return manifest
