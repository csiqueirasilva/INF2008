# DeepDRR-Inspired Rendering

This note records how we integrated the open-source DeepDRR toolkit into the
spine segmentation project. The goal is to close the appearance gap between
our simple pseudo-lateral projections (axis-aligned sums of HU values) and the
fluoroscopy frames seen in VFSS studies.

## Why DeepDRR?

The current pseudo-lateral generator collapses the CT volume along the left-
right axis and then applies ad-hoc tone tweaks. The result is high contrast,
“pitch black” images that bear little resemblance to X-ray fluoroscopy. DeepDRR
adds the missing physics:

- three-material decomposition (air / soft tissue / bone)
- poly-energetic forward projection (beam hardening)
- scatter estimation
- realistic noise injection

## Practical integration

DeepDRR’s full GPU projector requires `nvcc`, EGL, and a running X server. To
make progress quickly we:

1. vendored `libXrender.so` under `third_party/deepdrr/lib/linux` so the
   imported VTK modules load even on headless systems;
2. installed the upstream `deepdrr[cuda12x]` wheel inside the poetry
   environment (`poetry run pip install deepdrr[cuda12x]`);
3. implemented a *CPU* poly-energetic renderer that re-uses DeepDRR’s material
   segmentation (`Volume.segment_materials`) and spectral tables
   (`projector._get_spectrum`).

### Why not the CUDA projector (for now)?

The upstream `Projector` class compiles several custom CUDA kernels the first
time it runs. That compilation pipeline assumes:

- the `nvcc` binary from the CUDA toolkit is available on `PATH`, or via
  `CUPY_NVCC_PATH`/`NVCC` env vars, and
- the system has an X11 stack so VTK + PyOpenGL can load its GL backend.

In the course of integrating DeepDRR we ran into two blockers:

1. **No bundled `nvcc`.** Installing `deepdrr[cuda12x]` via pip brings in cuPy
   and the CUDA runtime libraries but *not* the actual `nvcc` compiler. Several
   NVIDIA wheels (`nvidia-cuda-nvcc-cu12`, `nvidia-cuda-toolkit`) exist, but
   they install only headers/support binaries (e.g. `ptxas`) and omit the
   `nvcc` executable. Without `nvcc`, cuPy falls back to NVRTC and still ends up
   shelling out to `nvcc`, yielding `AttributeError: 'NoneType' object has no
   attribute 'split'` when it tries to split the missing command string.
2. **Headless EGL only.** The machines available for this project lack a native
   X server. We vendored `libXrender` and forced `PYOPENGL_PLATFORM=egl` so VTK
   can start headless, but the CUDA kernels still fail to compile without `nvcc`.

Rather than block the entire exploration, we reproduced the physics path on the
CPU: segment air/soft/bone, integrate mass thickness per energy bin, apply
energy-dependent attenuation coefficients, and inject a lightweight Poisson
noise model. This yields fluoroscopy-like projections immediately, while keeping
the integration self-contained inside `deepdrr_bridge.py` (no external toolchain
needed).

### Roadmap for enabling the CUDA path

When someone is ready to wire up the full GPU projector, here’s the checklist:

1. **Provide a real CUDA toolkit.** Install the NVIDIA CUDA Toolkit matching
   the driver (e.g. 12.x) so that `nvcc` lives under `/usr/local/cuda/bin`. Set
   `NVCC` and `CUPY_NVCC_PATH` to point at that executable and ensure it’s on
   `PATH`.
2. **Retain the EGL tweaks.** Keep `PYOPENGL_PLATFORM=egl`,
   `PYGLET_HEADLESS=true`, and vendored `libXrender` so PyOpenGL + VTK can
   initialize in headless environments.
3. **Use DeepDRR’s projector.** Swap the CPU code in
   `render_deepdrr_projection` for a call to `deepdrr.Projector` with
   `scatter_num` set appropriately (the latest releases mark `add_scatter`
   deprecated). Once the kernel compilation succeeds, we can drop the custom
   numerical integration and inherit DeepDRR’s full scatter/noise models.

Documenting this limitation now will help a future contributor (or an AI agent)
hit the ground running when it’s time to enable the GPU version.

The renderer lives in `src/spine_segmentation/core/deepdrr_bridge.py`. It loads
a CT volume, segments the three materials, rotates/pans like the legacy
`build-pseudo-lateral` command (now deprecated), and integrates the mass thickness of each material
through the volume. For every energy bin in the DeepDRR spectrum we evaluate

```
I_E = exp(- Σ_material μ/ρ(E) · ρ_material · Δx)
```

The weighted sum over energies is converted to an 8-bit image (with optional
Poisson noise) so we can compare it directly to the legacy output.

## New CLI command

The command `spine deepdrr-project` renders a single view for inspection:

```bash
poetry run spine deepdrr-project \
  --ct data/CTSpine1K/raw_data/volumes/HNSCC-3DCT-RT/HN_P001.nii.gz \
  --out outputs/deepdrr/HNSCC-3DCT-RT_HN_P001_pitch90.png \
  --pitch 90
```

Key implementation details:

- Euler order matches `build-pseudo-lateral` (yaw → pitch → roll, degrees).
- Native resolution is preserved by default; pass `--no-native-resolution` (and
  optionally `--size`/`--sensor-width`) to force a square detector for
  comparisons with legacy outputs.
- Noise can be disabled via `--no-noise` for deterministic comparisons.

The example above produced
`outputs/deepdrr/HNSCC-3DCT-RT_HN_P001_pitch90.png`, which shows much smoother
attenuation than the legacy pseudo-lateral slab.

## Next steps

1. **Batch generation:** wire the bridge into `prepare-unet-tiles` so the new
   renderer can replace the current MIP-only projection during dataset builds.
2. **Material segmentation network:** swap thresholding for DeepDRR’s trained
   3D CNN once the checkpoint download is scripted (improves bone/soft-tissue
   separation).
3. **Scatter approximation:** the current CPU path omits scatter — investigate
   whether a light-weight blur + energy loss term produces a useful surrogate.
4. **Retraining & evaluation:** rebuild the UNet tiles with the new renderer,
   retrain, and re-run the VFSS batch inference to see if foreground detections
   rise above 1% of frames.

All steps, commands, and lingering assumptions are documented here so they can
feed directly into the Wednesday class presentation.
