# Attention Network for DeepFake Detection

A deepfake detector built for cross-dataset generalisation rather than in-domain accuracy. An Xception-based encoder–decoder produces both a classification embedding and a reconstruction of the input; the reconstruction residual then guides spatial attention, while a learnable Fourier-domain filter emphasises the spectral bands that carry forgery evidence.

M.Tech thesis, Delhi Technological University, 2022–2024. Trained on FaceForensics++.

### Relation to RECCE

The reconstruction–classification backbone follows [RECCE](https://openaccess.thecvf.com/content/CVPR2022/html/Cao_End-to-End_Reconstruction-Classification_Learning_for_Face_Forgery_Detection_CVPR_2022_paper.html) (Cao et al., CVPR 2022): the same Xception encoder, the same reconstruction decoder, and the same principle that reconstruction error localises manipulation.

Two changes:

- **RECCE's multi-scale graph reasoning module is removed.**
- **A learnable Fourier-domain filter branch is added**, taking the encoder features into frequency space, applying a learned per-bin mask, and returning to the spatial domain before fusion.

The intent was to replace graph-based spatial reasoning with spectral evidence, on the view that the resampling and blending traces deepfake pipelines leave are more directly visible in frequency space than in relational structure over spatial nodes.

This work led to a follow-on paper on deeper CLIP adaptation for deepfake detection, which reaches 0.949 average video-level AUROC across seven benchmarks. <!-- Once the preprint is live: See [Beyond Minimal Tuning](ARXIV_LINK). -->

---

## The problem this targets

Most deepfake detectors key on spatial RGB cues, or specialise for a particular condition such as heavy compression or low light. Both choices tend to fit the training manipulation rather than forgery in general, so in-domain accuracy looks excellent and cross-dataset accuracy collapses. That gap is what makes a detector unusable in practice, and it is the metric this project optimises for.

Two mechanisms address it here: reconstruction-guided attention, inherited from RECCE, and a learnable frequency filter, which is this project's addition.

## Why a frequency branch

Deepfake pipelines nearly all involve face alignment (warping), resizing, and affine transforms during blending. Interpolation kernels and repeated resampling leave structured irregularities in frequency space: energy bumps, ringing, aliasing patterns. Generative upsampling also distorts the spectral energy distribution, so a generated face often deviates from the roughly 1/f falloff that natural images follow, either by being too smooth or by carrying abnormal mid- and high-frequency energy. Blending boundaries at the jawline, hairline and cheeks inject extra high-frequency content at the seams.

These traces are subtle in RGB but structured in the spectrum, which is why a frequency branch adds something a spatial-only encoder misses.

The filter is **learned, not fixed**. A hand-designed high-pass filter commits in advance to which bands matter. Here the network decides, emphasising discriminative bands and suppressing brittle noisy ones.

## Why reconstruction-guided attention

The decoder reconstructs the input face from the encoder embedding. Where the model reconstructs poorly, something about the region is inconsistent with what the encoder learned to represent — which is exactly where manipulation artifacts tend to live.

So rather than letting attention learn where to look from scratch, the per-pixel reconstruction residual `|x − x̂|` is used to produce the attention map. Forgery evidence is not spread evenly across a face, and the residual gives the network a direct, unlearned signal about where to concentrate.

## Architecture

### Overall flow

```
input ──► encoder blocks 1–4 ──► embedding ──┬──► decoder 1–6 ──► reconstruction x̂
                                             │
                                             └──► encoder blocks 5–7 ──► Fourier filter ──► freq
                                                        │                                    │
                                                        └────► cross-modal attention ◄───────┘
                                                                        │
                                                                   + embedding  (residual)
                                                                        │
                                                            encoder block 8
                                                                        │
                            guided attention ◄── |x − x̂| ───────────────┘
                                                                        │
                                                  encoder blocks 9–12 ──► 1024
                                                     conv3 ──► 1536 ──► conv4 ──► 2048
                                                     global average pool ──► dropout ──► FC
```

### Encoder

Xception-style: depthwise separable convolutions with residual connections, batch norm throughout. Channel progression 3 → 32 → 64 → 128 → 256 → 728 (blocks 1–4), held at 728 through blocks 5–11, then 728 → 1024 → 1536 → 2048 before global pooling.

White noise is added to the input during training only.

### Decoder

Six blocks, each `UpsamplingNearest2d(scale=2) → SeparableConv2d → BatchNorm → ReLU`, running 728 → 256 → 256 → 128 → 128 → 64 → 3. The output is bilinearly interpolated back to input resolution to give the reconstruction.

### Fourier filter

Applied to the 728-channel feature map from encoder block 7, not to raw pixels.

1. `rfft2` per channel gives a complex spectrum of shape `[B, C, H, W/2+1]`. Only the non-redundant half is stored, since real inputs have conjugate-symmetric spectra.
2. Real and imaginary parts are stacked along the channel axis, because PyTorch `Conv2d` is real-valued.
3. A **grouped 1×1 convolution** (`in=2C, out=2C, groups=C`) followed by a sigmoid predicts the mask. 1×1 is deliberate: the convolution then acts independently per frequency bin `(u,v)`, mixing only the real and imaginary channels of the same feature channel. A 3×3 would mix neighbouring frequency bins, which smooths the mask rather than keeping it bin-wise and interpretable.
4. The mask gates the spectrum by element-wise multiplication.
5. `irfft2` returns a spatial map with certain bands emphasised or suppressed.

The current implementation predicts two real masks, one each for the real and imaginary parts, and applies them separately. This is a diagonal scaling rather than a full complex multiplication, so it changes amplitude per bin without rotating phase. That was a deliberate choice: allowing phase rotation degraded performance in experiments, which is consistent with phase being the less stable of the two components.

### Fusion and attention

The filtered frequency map and the spatial embedding are combined by cross-modal attention, with a residual connection back to the embedding:

```
fusion = CMA(embedding, freq) + embedding
```

Guided attention then operates on the result. It takes the input, the reconstruction and the embedding:

```python
residual = |x - x̂|                          # per-pixel reconstruction error, 3 channels
residual = interpolate(residual, embedding.shape[-2:])
res_map  = gated(residual)                  # Conv2d(3,3,3) → ReLU → Conv2d(3,1,1) → Sigmoid
out      = res_map * h(embedding) + dropout(embedding)
```

The gated block blends RGB reconstruction errors locally in a 3×3, then collapses them to a scalar importance per spatial location in a 1×1. `h` is a 1×1 projection with batch norm and ReLU that keeps the channel count but re-centres and rescales the features, so the multiplicative gate has predictable magnitude instead of being either toothless or overwhelming. The residual path means that when the gate zeroes a region, baseline information still reaches the classifier.

### Losses

Three terms are optimised jointly:

- **Classification.** Cross-entropy on the binary head.
- **Reconstruction.** Applied to the decoder output against the input face, which is what makes the residual meaningful as an attention signal.
- **Contrastive.** Computed from normalised embedding correlations collected at several points along the encoder and decoder, pulling real-face representations together so that manipulated faces stand out as outliers.

## Results

Trained on FaceForensics++ C23 and evaluated cross-dataset, so the model never saw these datasets during training. FF++ C40 appears in the table as a compression stress test, not as in-domain data.

| Dataset | AUROC |
|---|---|
| WildDeepfake | 0.802 |
| FF++ (C40) | 0.642 |
| Average across Celeb-DF / WildDeepfake / DFDC | 0.711 |

Cross-dataset numbers are the meaningful measure. In-domain performance on Celeb-DF v2 was high (98.1% accuracy, 0.998 AUC), but that figure largely reflects how learnable a single manipulation source is, not whether the detector generalises. The gap between the two motivated moving to a CLIP backbone in the follow-on work.

FF++ C40 is the weakest result, and the cause is structural: CRF 40 compression discards much of the high-frequency content the learnable filter depends on.

## Setup

```bash
git clone https://github.com/snehakumari1996/Attention_Network_for_Deepfake_Detection.git
cd Attention_Network_for_Deepfake_Detection
pip install -r requirements.txt
```

Developed with PyTorch on a single CUDA GPU. Exact package versions are pinned in `requirements.txt`.

## Data preparation

Training uses FaceForensics++ C23. Faces are detected, cropped and aligned to the encoder's input resolution before training, with real and fake frames kept in separate directories under a train/validation split.

Evaluation datasets (Celeb-DF v2, WildDeepfake, DFDC) are prepared the same way and used only at test time.

## Training

Training runs end to end: the classification, reconstruction and contrastive terms are optimised together rather than in stages. White noise is added to inputs during training only, as a regulariser against the model keying on clean-image statistics.

Hyperparameters live in the config file in this repository.

## Evaluation

Evaluation reports frame-level AUROC on each test dataset. Because the point of the model is cross-dataset behaviour, in-domain FF++ numbers are reported for reference only.

Pretrained weights are available on request.

## Limitations

- Trained on a single manipulation source, so performance drops on unseen generators.
- Heavy compression attenuates the bands the Fourier filter relies on, visible in the FF++ C40 result.
- The filter applies diagonal scaling rather than full complex multiplication, so it cannot adjust phase. Phase rotation was tested and degraded results, but a magnitude–phase parameterisation with a bounded gain remains untested.
- Predates diffusion-based face generation; not evaluated on diffusion-generated content.
- No evaluation across demographic groups. The benchmarks lack annotations for skin tone, gender and age, so any disparity in error rates is unmeasured.

## Citation

```bibtex
@mastersthesis{kumari2024attention,
  title  = {Attention Network for DeepFake Detection},
  author = {Kumari, Sneha},
  school = {Delhi Technological University},
  year   = {2024}
}
```

## Contact

Sneha Kumari — sneha.k.1996@gmail.com
