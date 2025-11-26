# Preprocess Compare: Two-Step Neck → Spine Refinement

This note captures a practical two-step pipeline to get closer to the VFSS reference look before training:

## Why two steps?
- The full-frame DRR binarization is still fusing vertebrae into a single blob.
- A coarse, neck-only crop reduces background variation and lets us tune histogram/contrast just for the spine.

## Step 1 — Coarse neck mask + crop
Goal: detectar o pescoço completo, recortar e produzir o input do passo 2. Este passo define o “input de rede” (imagens tipo referência binarizadas) e o “input de treino” (imagens sintéticas/rotuladas binarizadas) com o mesmo pipeline de limiarização.

Entrada:
- Imagens de referência (ex.: `data/frames/50/*.png`) passando pela nossa binarização (Otsu após um leve blur).
- Imagens sintéticas DeepDRR com labels de vértebras (1–7) para derivar o bbox do pescoço a partir da máscara de vértebras já disponível.

Saída:
- `binary_pair.png` (Otsu dos dois lados) para inspecionar a qualidade.
- BBox do pescoço das vértebras (mínimo retângulo que cobre labels 1–7) para recortar depois; pode ser salvo como coords ou gerar a imagem já recortada.

Recipe (exemplo isolado):
```
poetry run spine deepdrr-pair \
  --ct data/CTSpine1K/raw_data/volumes/HNSCC-3DCT-RT/HN_P001.nii.gz \
  --label-ct data/CTSpine1K/raw_data/labels/HNSCC-3DCT-RT/HN_P001_seg.nii.gz \
  --frame data/frames/50/v50_f145.png \
  --yaw 180 --pitch 0 --roll 180 \
  --size 384 --sensor-width 384 --pixel-mm 0.7 --sdd 1400 \
  --guide-labels 1-7 --no-guides \
  --crop-square --match-frame-size \
  --aperture --aperture-radius-scale 0.5 --aperture-softness 0.0 --aperture-blur 0 \
  --aperture-inside 1.0 --aperture-outside 0.0 \
  --zoom-factor 1.3 --pan-x-px -75 --pan-y-px 30 \
  --no-noise \
  --spectrum 60KV_AL35 \
  --bone-scale 1.35 \
  --no-hist-match --no-edge-enhance --no-clahe \
  --blur-kernel 5 \
  --out-dir outputs/pair_compare_coarse
```

Para rodar em todo o conjunto de referência (`data/frames/50`) e inspecionar:
```
for f in data/frames/50/*.png; do
  base=$(basename "$f" .png)
  poetry run spine deepdrr-pair \
    --ct data/CTSpine1K/raw_data/volumes/HNSCC-3DCT-RT/HN_P001.nii.gz \
    --label-ct data/CTSpine1K/raw_data/labels/HNSCC-3DCT-RT/HN_P001_seg.nii.gz \
    --frame "$f" \
    --yaw 180 --pitch 0 --roll 180 \
    --size 384 --sensor-width 384 --pixel-mm 0.7 --sdd 1400 \
    --guide-labels 1-7 --no-guides \
    --crop-square --match-frame-size \
    --aperture --aperture-radius-scale 0.5 --aperture-softness 0.0 --aperture-blur 0 \
    --aperture-inside 1.0 --aperture-outside 0.0 \
    --zoom-factor 1.3 --pan-x-px -75 --pan-y-px 30 \
    --no-noise \
    --spectrum 60KV_AL35 \
    --bone-scale 1.35 \
    --no-hist-match --no-edge-enhance --no-clahe \
    --blur-kernel 5 \
    --save-bbox --crop-to-bbox --crop-margin 0.05 \
    --out-dir "outputs/step1_coarse/${base}"
done
```
Inspecione `outputs/step1_coarse/*/combined/binary_pair.png` e extraia o bbox das vértebras pelo label map (IDs 1–7) para recortar (salvar coords ou imagem recortada).

## Step 2 — Fine spine contrast on the crop
Goal: enhance vertebra edges inside the cropped region without blowing highlights.

After cropping, re-run contrast on the crop (or re-render at higher zoom centered on the crop) with a gentle edge and histogram lift:
```
poetry run spine deepdrr-pair \
  --ct ... --label-ct ... --frame ... \
  --yaw 180 --pitch 0 --roll 180 \
  --size 384 --sensor-width 384 --pixel-mm 0.7 --sdd 1400 \
  --guide-labels 1-7 --no-guides \
  --crop-square --match-frame-size \
  --save-bbox --crop-to-bbox --crop-margin 0.05 \
  --aperture --aperture-radius-scale 0.5 --aperture-softness 0.0 --aperture-blur 0 \
  --aperture-inside 1.0 --aperture-outside 0.0 \
  --zoom-factor 1.5 --pan-x-px -50 --pan-y-px 20 \
  --no-noise \
  --spectrum 60KV_AL35 \
  --bone-scale 1.35 \
  --edge-enhance --edge-sigma 1.0 --edge-amount 0.8 \
  --clahe --clip-limit1 1.2 --clip-limit2 1.2 --tile-size1 12 --tile-size2 12 \
  --blur-kernel 3 \
  --no-hist-match \
  --out-dir outputs/pair_compare_fine
```
Inspect `combined/clahe2_pair.png` and `combined/binary_pair.png` to ensure vertebrae remain separated.

## Notes and rationale
- **Energy preset**: only three spectra exist (60KV_AL35, 90KV_AL40, 120KV_AL43). 60 kV maximizes bone contrast.
- **Bone scale**: multiplier on bone attenuation; start ~1.3–1.35 to brighten bone without saturating.
- **Noise/scatter**: we are not simulating scatter; `--no-noise` removes Poisson grain.
- **Blur before Otsu**: raising `--blur-kernel` (e.g., 5) stabilizes the mask and reduces speckle.
- **Edge enhance**: keep mild to avoid ringing in the binary mask; sigma ~1.0, amount <1.0.
- **CLAHE**: gentle clip limits (≈1.2) prevent speckle amplification; adjust up/down if needed.

## Next steps
- Automate the crop: derive the neck bounding box from the Otsu mask or projected labels and re-render at a tighter FOV/zoom.
- If contrast gap persists, consider histogram matching on the crop only, or try a gradient/edge map input for training.
- For style gap, a later option is a CycleGAN (synthetic→VFSS) while keeping labels aligned.

## Pipeline de etapas e comandos (proposta)
Para não perder o fio, dividimos o passo 1 em subcomandos/produtos claros:

1) Preparar imagens head/neck com fatiamento e metadados (labels)
   - Entrada: todos os casos de head/neck (CT e labels 3D; frames de referência).
   - Ação: gerar projeções/pseudolaterais com labels 2D e salvar em um diretório de inspeção (ex.: `prepared/headneck/`), mantendo metadados (IDs de vértebras, resolução).
   - Saída: imagens 2D (projeção + máscara 2D) e JSON/CSV com labels 2D (centroides/contornos) para uso no bbox.

2) Treinar um detector de bbox de pescoço
   - Entrada: imagens (referência e sintéticas) + bbox derivado das labels de vértebras (1–7).
   - Ação: treinar uma rede simples (ex.: RetinaNet/Tiny-YOLO ou mesmo um regressor de bbox) para retornar bbox do pescoço em imagens do estilo de referência.
   - Saída: pesos do detector e um comando `detect-neck-bbox` que salva bbox em JSON/CSV por imagem.

3) Crop das imagens e ajuste dos metadados
   - Entrada: imagens + bbox (JSON/CSV) + labels 2D.
   - Ação: recortar as imagens segundo bbox; ajustar coordenadas 2D dos segmentos para o sistema do crop; salvar no diretório `crops/`.
   - Saída: imagens recortadas + metadados/labels atualizados.

4) Suavização/realce preservando vértebras
   - Entrada: crops.
   - Ação: aplicar pipeline leve (ex.: blur leve + CLAHE suave + unsharp moderado) para realçar bordas sem perder topologia dos segmentos; gerar variações (ex.: “clahe”, “edge-lite”).
   - Saída: versões tratadas em `crops_processed/` para treino.

5) Treino de segmentação nos crops
   - Entrada: crops processados + labels ajustados.
   - Ação: treinar U-Net/DeepLab focando no pescoço recortado.
   - Saída: pesos do segmentador específico de pescoço.

6) Inferência encadeada (bbox → crop → segmentação)
   - Entrada: imagem de referência (ex.: `data/frames/50/*.png`).
   - Ação: (a) detector retorna bbox e salva JSON/CSV; (b) comando faz crop; (c) segmentador infere e gera: (i) imagem recortada, (ii) imagem original com labels pintados na posição original, (iii) labels 2D em JSON/CSV.
   - Saída: diretório com `crop.png`, `overlay.png`, `labels.json/csv`, possibilitando inspeção rápida.

Documentação e inspeção:
 - Manter todos os intermediários (`binary_pair.png`, `clahe_pair`, `overlay_pair`) para depuração.
 - Para `data/frames/50`, rodar um loop que gera crops e bboxes para todas as imagens e revisar se o pescoço foi corretamente extraído.
