# src/spine_segmentation/core/drr_utils.py
import numpy as np
import torch
from pathlib import Path
from diffdrr.drr import DRR
from diffdrr.data import read as drr_read

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _ensure_labelmap_key(subj):
    if "labelmap" in subj:
        return subj
    for k in ("label", "labels", "seg", "mask"):
        if k in subj:
            subj["labelmap"] = subj[k]
            break
    return subj

def make_subject_from_paths(ct_path, seg_path=None, mask_volume=False, orientation=None):
    # Let diffdrr infer orientation; avoid hardcoding "RAS"/"LPS"
    subj = drr_read(volume=str(ct_path),
                    labelmap=(str(seg_path) if seg_path else None),
                    orientation=orientation)
    subj = _ensure_labelmap_key(subj)
    if mask_volume and seg_path is not None and "labelmap" in subj:
        vol = subj["volume"].data
        lab = subj["labelmap"].data
        mask = (lab > 0).to(vol.dtype)
        subj["volume"].data = vol * mask
    return subj

def make_drr(subject, sdd=1600.0, height=512, delx=1.2, dev=None):
    if dev is None: dev = device()
    return DRR(subject, sdd=sdd, height=height, delx=delx).to(dev)

def _auto_delx_from_subject(subject, height, safety=1.15, prefer_label=True):
    """
    Pick pixel size (mm/pixel) so the detector height covers a 'safe' extent.
    Uses label bbox if available; falls back to whole volume.
    """
    # spacing order in TorchIO is (sx, sy, sz) in mm
    spx, spy, spz = subject["volume"].spacing
    shp = subject["volume"].data.shape[-3:]  # (X,Y,Z)
    if prefer_label and "labelmap" in subject:
        lab = subject["labelmap"].data[0].detach().cpu().numpy()
        where = np.argwhere(lab > 0)
        if where.size > 0:
            xmin,ymin,zmin = where.min(0)
            xmax,ymax,zmax = where.max(0) + 1
            dims_mm = np.array([(xmax-xmin)*spx, (ymax-ymin)*spy, (zmax-zmin)*spz], dtype=float)
        else:
            dims_mm = np.array([shp[0]*spx, shp[1]*spy, shp[2]*spz], dtype=float)
    else:
        dims_mm = np.array([shp[0]*spx, shp[1]*spy, shp[2]*spz], dtype=float)

    # Use diagonal as a conservative bound so any yaw/pitch fits
    diag_mm = float(np.linalg.norm(dims_mm))
    # delx so that detector height covers the diagonal comfortably
    return safety * diag_mm / max(1, int(height))

def render_drr(subject, yaw=0.0, pitch=0.0, roll=0.0,
               tx=0.0, ty=None, tz=0.0,
               sdd=1600.0, height=512, delx=1.2, dev=None,
               auto_fov=False):
    dev = dev or device()
    if auto_fov:
        delx = _auto_delx_from_subject(subject, height)

    if ty is None:
        # place camera reasonably far so magnification isn’t crazy
        ty = 0.9 * sdd

    drr = make_drr(subject, sdd=sdd, height=height, delx=delx, dev=dev)
    rot = torch.tensor([[float(yaw), float(pitch), float(roll)]], device=dev, dtype=torch.float32)
    trs = torch.tensor([[float(tx),  float(ty),    float(tz)   ]], device=dev, dtype=torch.float32)
    with torch.no_grad():
        arr = drr(rot, trs, parameterization="euler_angles", convention="ZXY")[0]
        arr = arr.detach().cpu().numpy()
    arr = np.squeeze(arr)                        # (H,W)
    u8  = (255 * (arr - arr.min()) / (arr.ptp() + 1e-6)).astype(np.uint8)
    return np.ascontiguousarray(u8)

def project_binary_mask(subject_ct, seg_path, rot, trs,
                        sdd=1600.0, height=512, delx=1.2, dev=None, thresh=0.001):
    dev = dev or device()
    seg_subject = make_subject_from_paths(seg_path, seg_path=None, mask_volume=False)
    drr_seg = make_drr(seg_subject, sdd=sdd, height=height, delx=delx, dev=dev)
    rot = rot.to(dtype=torch.float32); trs = trs.to(dtype=torch.float32)
    with torch.no_grad():
        thick = drr_seg(rot, trs, parameterization="euler_angles", convention="ZXY")[0]
        thick = thick.detach().cpu().numpy()
    thick = np.squeeze(thick)
    t = max(thresh, 0.10 * (np.nanmax(thick) if np.isfinite(thick).all() else 1.0))
    return ((thick > t).astype(np.uint8) * 255)

def mask_to_outline(mask_u8, k=3):
    import cv2
    kernel = np.ones((k,k), np.uint8)
    er = cv2.erode(mask_u8, kernel, iterations=1)
    edge = cv2.subtract(mask_u8, er)
    return edge

def drr_edges(drr_u8):
    """Robust edges from a (H,W) DRR image."""
    import cv2
    if drr_u8.ndim == 3:
        drr_u8 = drr_u8[..., 0]
    if drr_u8.std() < 2:  # nearly flat
        return np.zeros_like(drr_u8)
    eq = cv2.equalizeHist(drr_u8)
    med = np.median(eq)
    lo = max(0, 0.66 * med)
    hi = min(255, 1.33 * med)
    return cv2.Canny(eq, lo, hi)
