"""Utilities to generate DeepDRR projections for CTSpine1K volumes.

This module wraps the `deepdrr` library so we can experiment with the
poly-energetic / scatter-aware renderer without dragging the rest of the
pipeline into its heavy dependency stack.  The helper ensures:

* Required shared libraries (e.g. libXrender) are discoverable at runtime.
* deepdrr is imported in headless mode (EGL backend, no X server needed).
* A minimal `SimpleDevice` camera is configured to mimic the geometry used
  by our existing pseudo-lateral generator (sagittal projection with rays
  marching roughly along the +X axis after the requested yaw/pitch/roll
  rotations are applied).

The entry point `render_deepdrr_projection` returns a single-channel numpy
array (uint8) suitable for quick inspection or drop-in comparison against
our legacy DRR output.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

# Path to the vendored libXrender copy (avoids having to apt-install system
# packages on machines where we lack sudo).  The dynamic linker only updates
# its search path per-process, so this helper must run before importing
# deepdrr and/or vtk/pyrender.
_LIB_DIR = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "deepdrr"
    / "lib"
    / "linux"
)


def _prepend_library_path(path: Path) -> None:
    """Prepend `path` to LD_LIBRARY_PATH for the current process."""

    if not path.exists():
        raise FileNotFoundError(
            f"Expected shared library directory not found: {path}"
        )

    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [str(path)] + ([current] if current else [])
    os.environ["LD_LIBRARY_PATH"] = ":".join(entries)


def _prepare_environment() -> None:
    """Set environment variables before importing heavy GPU/toolkit deps."""

    # Ensure EGL-based OpenGL context (no X server needed).
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    os.environ.setdefault("DEEPDRR_HEADLESS", "1")
    os.environ.setdefault("CUPY_CUDA_COMPILER", "nvrtc")

    # Guarantee libXrender exists on the search path before loading vtk/pyrender.
    if "libXrender" not in os.environ.get("LD_PRELOAD", ""):
        _prepend_library_path(_LIB_DIR)

    # Prepend nvcc path from the pip-provided CUDA toolkit (if present).
    try:
        import site

        for base in site.getsitepackages():
            candidate = Path(base) / "nvidia" / "cuda_nvcc" / "bin"
            nvcc = candidate / "nvcc"
            if nvcc.exists():
                os.environ.setdefault("NVCC", str(nvcc))
                os.environ.setdefault("CUPY_NVCC_PATH", str(nvcc))
                path_entries = [str(candidate)] + os.environ.get("PATH", "").split(":")
                os.environ["PATH"] = ":".join([entry for entry in path_entries if entry])
                break
    except Exception:
        # Non-fatal; deepdrr will fail later if nvcc is absolutely required.
        pass

    # Proactively load libXrender so subsequent imports can locate it even if
    # the dynamic loader ignores our LD_LIBRARY_PATH tweak.
    import ctypes

    lib_candidate = _LIB_DIR / "libXrender.so.1"
    if not lib_candidate.exists():
        lib_candidate = _LIB_DIR / "libXrender.so.1.3.0"
    if not lib_candidate.exists():
        raise FileNotFoundError("libXrender shared object not found in vendored directory")
    try:
        ctypes.cdll.LoadLibrary(str(lib_candidate))
    except OSError as exc:
        raise RuntimeError(f"Failed to load {lib_candidate}: {exc}") from exc


def _as_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return ZYX Euler rotation matrix (deg) matching build_pseudo_lateral."""

    rz = math.radians(float(yaw))
    ry = math.radians(float(pitch))
    rx = math.radians(float(roll))

    cz, sz = math.cos(rz), math.sin(rz)
    cy, sy = math.cos(ry), math.sin(ry)
    cx, sx = math.cos(rx), math.sin(rx)

    rz_mat = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry_mat = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx_mat = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz_mat @ ry_mat @ rx_mat


def _rotate_volume(volume, yaw: float, pitch: float, roll: float) -> None:
    """Apply in-place world rotation matching the legacy pseudo-lateral view."""

    from deepdrr import geo

    rot_mat = _as_rotation_matrix(yaw, pitch, roll)
    rotation = geo.FrameTransform.from_rotation(rot_mat)
    volume.rotate(rotation)


def render_deepdrr_projection(
    ct_path: Path,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    sensor_size_px: int = 512,
    sensor_width_px: int | None = None,
    native_resolution: bool = False,
    pixel_size_mm: float = 1.2,
    source_to_detector_distance_mm: float = 1600.0,
    add_noise: bool = True,
    spectrum: str = "90KV_AL40",
    tone_style: str = "smooth",
    apply_clahe: bool = False,
    bone_scale: float = 1.0,
    slice_offset_mm: float = 0.0,
    slice_thickness_mm: float = 0.0,
) -> np.ndarray:
    """Generate a single DeepDRR projection for the provided CT volume.

    Args:
        ct_path: Path to a 3D CT volume (NIfTI) in Hounsfield units.
        yaw/pitch/roll: Euler angles (deg) applied in the same order as the
            legacy generator (Z → Y → X).
        sensor_size_px: Detector height in pixels. When ``sensor_width_px`` is ``None`` the
            same value is used for the width.
        sensor_width_px: Optional detector width in pixels. If omitted, a square detector is
            assumed.
        native_resolution: If True, skip resizing so the returned image stays on the grid defined
            by the CT-derived projection.
        pixel_size_mm: Detector pixel pitch (mm/pixel).
        source_to_detector_distance_mm: Distance (mm) from X-ray source to
            detector plane.
        add_scatter / add_noise: Toggle DeepDRR's learned scatter model and
            Poisson/read-out noise injection.
        spectrum: Named poly-energetic spectrum to use (DeepDRR includes
            presets such as "90KV_AL40").
        tone_style: Either "smooth" (default gamma lift) or "raw" to return
            the uninverted attenuation map.
        apply_clahe: If True, apply CLAHE after tone mapping to boost local
            contrast.
        bone_scale: Multiplier applied to bone attenuation coefficients to
            boost bone contrast (e.g., 1.2 makes bone brighter).
        slice_offset_mm: Offset (mm) along the projection axis. If slice_thickness_mm <= 0,
            this translates the whole volume before projection. If slice_thickness_mm > 0,
            it shifts the thin-slab center (positive moves toward +X after rotations).
        slice_thickness_mm: Slab thickness (mm) summed along the projection axis. If <=0,
            uses the full volume (legacy behavior).

    Returns:
        uint8 numpy array shaped (H, W) with values stretched to [0, 255]. The
        output matches ``sensor_size_px``/``sensor_width_px`` unless
        ``native_resolution`` is requested.
    """

    _prepare_environment()

    from deepdrr import Volume, geo
    from deepdrr.material import Material
    from deepdrr.projector.projector import _get_spectrum

    ct_path = Path(ct_path)
    if not ct_path.exists():
        raise FileNotFoundError(f"CT volume not found: {ct_path}")

    nii = nib.load(str(ct_path))
    hu = np.asarray(nii.get_fdata(), dtype=np.float32)
    spacing = np.asarray(nii.header.get_zooms()[:3], dtype=np.float32)

    # Reorder to (Z, Y, X) to match the legacy pseudo-lateral code.
    hu = np.transpose(hu, (2, 1, 0))
    spacing = spacing[[2, 1, 0]]

    density = Volume._convert_hounsfield_to_density(hu)
    anatomical_from_IJK = geo.FrameTransform(nii.affine)
    materials = Volume.segment_materials(
        np.transpose(hu, (2, 1, 0)),  # back to nib orientation for the helper
        anatomical_from_IJK,
        use_thresholding=True,
        use_cached=False,
        save_cache=False,
        cache_dir=None,
        cache_name=ct_path.stem,
    )
    material_masks = {
        name: np.transpose(mask.astype(np.float32), (2, 1, 0))
        for name, mask in materials.items()
    }

    def rotate(arr: np.ndarray, order: int) -> np.ndarray:
        out = arr
        if yaw and abs(float(yaw)) > 1e-3:
            out = ndi.rotate(out, angle=float(yaw), axes=(1, 2), reshape=True, order=order, mode="nearest")
        if pitch and abs(float(pitch)) > 1e-3:
            out = ndi.rotate(out, angle=float(pitch), axes=(0, 2), reshape=True, order=order, mode="nearest")
        if roll and abs(float(roll)) > 1e-3:
            out = ndi.rotate(out, angle=float(roll), axes=(0, 1), reshape=True, order=order, mode="nearest")
        return out

    density = rotate(density, order=1)
    for key in list(material_masks.keys()):
        material_masks[key] = rotate(material_masks[key], order=0)

    # Project along +X (axis=2) to mimic lateral view.
    proj_axis = 2
    spacing_axis_mm = float(spacing[proj_axis])
    spacing_axis_cm = spacing_axis_mm / 10.0  # convert mm → cm

    use_slab = slice_thickness_mm is not None and slice_thickness_mm > 0.0
    offset_mm = slice_offset_mm if slice_offset_mm is not None else 0.0
    if use_slab:
        vox_thick = max(1, int(round(slice_thickness_mm / max(spacing_axis_mm, 1e-6))))
        offset_vox = int(round(offset_mm / max(spacing_axis_mm, 1e-6)))
        center = int(np.clip(density.shape[proj_axis] // 2 + offset_vox, 0, density.shape[proj_axis] - 1))
        start = max(0, min(center - vox_thick // 2, density.shape[proj_axis] - vox_thick))
        end = start + vox_thick
    else:
        start, end = 0, density.shape[proj_axis]
        if abs(offset_mm) > 1e-6:
            shift_vox = float(offset_mm / max(spacing_axis_mm, 1e-6))
            density = ndi.shift(density, shift=[0, 0, shift_vox], order=1, mode="constant", cval=0.0)
            for key in list(material_masks.keys()):
                material_masks[key] = ndi.shift(
                    material_masks[key], shift=[0, 0, shift_vox], order=0, mode="constant", cval=0.0
                )

    # Ensure masks align with density shape (due to reshape=True rotations).
    target_shape = density.shape
    for key, mask in material_masks.items():
        if mask.shape != target_shape:
            # Pad/crop to match shape (nearest semantics)
            pad = [(0, max(0, t - s)) for s, t in zip(mask.shape, target_shape)]
            mask = np.pad(mask, pad, mode="constant")
            slices = tuple(slice(0, t) for t in target_shape)
            mask = mask[slices]
        material_masks[key] = mask.astype(np.float32)

    mass_thickness = {}
    for name, mask in material_masks.items():
        if name == "air":
            continue  # ignore air contribution
        if use_slab:
            slicer = [slice(None)] * 3
            slicer[proj_axis] = slice(start, end)
            mass = (density * mask)[tuple(slicer)].sum(axis=proj_axis) * spacing_axis_cm
        else:
            mass = (density * mask).sum(axis=proj_axis) * spacing_axis_cm
        mass_thickness[name] = mass

    if not mass_thickness:
        raise RuntimeError("Material segmentation returned no non-air voxels")

    spectrum_arr = _get_spectrum(spectrum)
    energies_keV = spectrum_arr[:, 0] / 1000.0
    weights = spectrum_arr[:, 1]
    weights = weights / weights.sum()

    # Pre-fetch materials
    material_objs = {name: Material.from_string(name) for name in mass_thickness.keys()}

    attenuation = None
    image = np.zeros_like(next(iter(mass_thickness.values())), dtype=np.float32)

    for energy_keV, weight in zip(energies_keV, weights):
        total = np.zeros_like(image)
        for name, mass in mass_thickness.items():
            coeff = material_objs[name].get_coefficients(energy_keV).mu_over_rho
            if name.lower().startswith("bone") or "bone" in name.lower():
                coeff = coeff * float(bone_scale)
            total += coeff * mass
        image += weight * np.exp(-total)

    # Optional simple noise model (Poisson) to mimic fluoroscopy grain.
    if add_noise:
        photons = np.random.poisson(lam=np.clip(image * 4e4, 1e2, 5e5))
        image = photons.astype(np.float32) / photons.max()

    image -= image.min()
    if image.max() > 0:
        image /= image.max()

    import cv2

    if tone_style.lower() == "smooth":
        gamma = 0.5
        toned = np.power(np.clip(image, 0.0, 1.0), gamma)
        image_u8 = np.clip(toned * 255.0, 0, 255).astype(np.uint8)
        # image_u8 = cv2.GaussianBlur(image_u8, (3, 3), 0)  # comment this line to remove Gaussian blur
    else:
        image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    if not native_resolution:
        target_h = int(sensor_size_px)
        target_w = int(sensor_width_px if sensor_width_px is not None else sensor_size_px)
        if target_h <= 0 or target_w <= 0:
            raise ValueError("Detector dimensions must be positive when native_resolution is False")
        if image_u8.shape[0] != target_h or image_u8.shape[1] != target_w:
            image_u8 = cv2.resize(image_u8, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_u8 = clahe.apply(image_u8)

    return np.ascontiguousarray(image_u8)


__all__ = ["render_deepdrr_projection"]
