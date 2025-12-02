## Notes: Bounding Box Inflation from Fragmented Vertebra Masks

### Symptom
- Neck “union” bounding boxes (used for cropping) were oversized and misaligned relative to the visible spine in synthetic DRR slices (e.g., `HN_P007/off_m3p000`).
- Overlays showed multiple disjoint blobs per vertebra label; union bbox encompassed distant specks, producing a too-large crop.

### Root Cause
- Thin-slab DeepDRR projections produce sparse/fragmented vertebra masks, especially when using guide-label subsets.
- Mild denoising (3×3 opening, 1 iter) left multiple connected components per class (e.g., C3 had 3 components; some areas as small as ~50 px).
- Union bbox was computed over all surviving pixels; stray components inflated width/height.

### Mitigation Applied
- Stronger morphological opening (erode→dilate) per class and largest-component retention:
  - Kernel: 9×9 ellipse, iterations: 2
  - Drop class if largest component area < 500 px (tunable)
- Priority law: enforce single class per pixel (C1>C2>…>C7) before bbox computation.
- Recompute per-class bboxes and union neck bbox from the cleaned mask.
- Result: union bbox tight to spine; per-vertebra boxes correspond to dominant component only.

### Why Opening + Largest Component
- Opening removes thin bridges and small blobs (Gonzalez & Woods, *Digital Image Processing*, ch. 9).
- Largest-component selection is a standard postprocess in medical segmentation to suppress speckles (e.g., Kamnitsas et al., MICCAI 2017 “EMMA” uses CC filtering).
- Combined, this minimizes bounding-box inflation due to outlier pixels.

### Implementation (key steps)
- Cleaner (`scripts/clean_bitmasks_priority.py`):
  - Optional copy to `out-root` to preserve originals.
  - Per-class: morphological opening (ksize, iters), keep only the largest CC (optional min-area), priority packing to absolute bits.
  - Writes overlays matched to mask type (clahe2 vs. circular-synth).
- Aggressive clean prototype for the failing slice:
  - Kernel 9×9, iterations 2, min-area 500.
  - Recomputed union bbox and per-class bboxes; neck crop now matches spine extent.

### Outstanding Considerations
- The optimal kernel/min-area are slab/thickness-dependent; may need tuning per dataset (thin vs. thick slabs).
- If labels are inherently fragmented (true anatomy vs. mask noise), over-aggressive cleaning could truncate anatomy. Monitor with overlays.
- For production, batch-apply tuned params and rebuild crops; keep originals for traceability.

### References
- Gonzalez, Woods. *Digital Image Processing*, 4th ed. (Morphological opening; noise suppression.)
- Kamnitsas et al. “Efficient Multi-Scale 3D CNN with Fully Connected CRF for Accurate Brain Lesion Segmentation.” MICCAI 2017. (Use of connected components to filter small spurious regions.)
- Milletari et al. “V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.” 2016. (Common practice of CC filtering in postprocessing.)
