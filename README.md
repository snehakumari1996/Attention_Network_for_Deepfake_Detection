# Attention Network for DeepFake Detection

Reconstruction-classification learning with a learnable spectral filter, for face forgery detection that generalizes across datasets.

M.Tech thesis, Delhi Technological University, 2022 to 2024. Implemented in PyTorch.

## Introduction

Face forgery detectors tend to learn artifacts specific to the manipulations in their training set. They score well in domain and transfer poorly to unseen generators, which is the failure that matters for deployment. This repository contains a detector built to be evaluated cross-dataset from the start.

The architecture follows the reconstruction-classification framework of RECCE [1]. An Xception encoder produces a latent embedding, a decoder reconstructs the input face from it, and the per-pixel reconstruction residual is used to gate attention: regions the model reconstructs poorly are regions inconsistent with its representation of authentic faces.

Two changes are made to that framework. The multi-scale graph reasoning module is removed. In its place, encoder features are transformed into the Fourier domain, modulated by a learned per-bin mask, and returned to the spatial domain before fusion with the embedding. The reasoning is that the resampling and blending traces left by forgery pipelines are expressed more directly in frequency space than in relational structure over spatial nodes.

## Results

All models are trained on FaceForensics++ C23 and evaluated without fine-tuning. Frame-level AUROC.

| Method | Celeb-DF v2 | WildDeepfake | DFDC | Mean |
|---|---|---|---|---|
| RECCE [1] | 0.687 | 0.643 | 0.691 | 0.674 |
| This work | see note | 0.802 | see note | 0.711 |

The spectral branch gives its largest gain on WildDeepfake, which is drawn from in-the-wild footage with heterogeneous provenance and compression. This is consistent with the branch keying on resampling traces rather than on manipulation-specific texture.

Under heavy compression the picture reverses. On FF++ C40 the model reaches 0.642 AUROC, its weakest result, because CRF 40 encoding discards much of the high frequency content the filter operates on.

In-domain performance on Celeb-DF v2 reaches 98.1% accuracy and 0.998 AUC. That number reflects how learnable a single manipulation source is and should not be read as evidence of generalization. The gap between it and the cross-dataset figures above is what motivated the follow-on work on vision-language backbones.

RECCE figures are as reported in [7].

<!-- TODO before publishing: fill in per-dataset Celeb-DF and DFDC AUROC in the table above.
     The mean of 0.711 is already reported, so the individual numbers should be recoverable
     from the evaluation logs. A table with gaps invites more doubt than a lower number would. -->

## Method

The full derivation is in the thesis. A summary of the two components follows.

**Spectral filter.** The filter acts on the 728-channel feature map from encoder block 7, not on raw pixels. A real-input 2D FFT (`rfft2`) moves the features to frequency space, retaining the non-redundant half of the conjugate-symmetric spectrum. Because PyTorch convolutions are real-valued, real and imaginary components are stacked along the channel axis, and a grouped 1x1 convolution followed by a sigmoid predicts the mask. Grouping means each feature channel learns its own transform over its real and imaginary pair. The 1x1 kernel is deliberate: at this point a spatial location is a frequency bin, so a 1x1 acts per bin, whereas a 3x3 would mix neighbouring frequencies and smooth the mask. An inverse transform returns the filtered features to the spatial domain.

The implementation predicts two real masks and applies them separately to the real and imaginary parts. This is diagonal scaling rather than full complex multiplication: it adjusts per-bin amplitude but cannot rotate phase. A full complex mask was tested and degraded cross-dataset performance, which is consistent with phase being the less stable component under compression and resampling.

**Reconstruction-guided attention.** Filtered spectral features are fused with the spatial embedding by cross-modal attention with a residual connection. The absolute reconstruction residual is then passed through a small convolutional block and a sigmoid to produce a single-channel spatial attention map, which gates a projected copy of the fused features. An additive residual path preserves baseline information where the gate suppresses a region.

**Training.** Classification, reconstruction and a contrastive term over normalized embedding correlations are optimized jointly, end to end. White noise is added to inputs during training only.

## Requirements

<!-- TODO: replace with the actual versions from your environment.
     Run: pip freeze | grep -iE "torch|torchvision|albumentations|timm|numpy|scipy|pyyaml"
     RECCE pins Pytorch 1.7.1, Torchvision 0.8.2, Albumentations 1.0.3, Timm 0.3.4,
     TensorboardX 2.1, Scipy 1.5.2, PyYaml 5.3.1, which is a reasonable reference point. -->

```bash
git clone https://github.com/snehakumari1996/Attention_Network_for_Deepfake_Detection.git
cd Attention_Network_for_Deepfake_Detection
pip install -r requirements.txt
```

## Dataset preparation

Four datasets are used. Training uses FaceForensics++ C23; the others are held out for evaluation.

- [FaceForensics++](https://github.com/ondyari/FaceForensics) [2]
- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) [3]
- [WildDeepfake](https://github.com/deepfakeinthewild/deepfake-in-the-wild) [4]
- [DFDC](https://ai.meta.com/datasets/dfdc/) [5]

The originals are video. Facial crops are extracted per frame before training, using [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) [6], and stored with authentic and manipulated frames in separate directories.

<!-- TODO: state the extracted crop resolution and frames sampled per video. -->

## Training

<!-- TODO: replace with the exact command this repository uses. -->

```bash
python train.py --config config/default.yaml
```

Training parameters are set in the config file: batch size, learning rate, optimizer, schedule, and the weights on the reconstruction and contrastive terms.

## Testing

<!-- TODO: replace with the exact command this repository uses. -->

```bash
python test.py --config config/default.yaml
```

Reports frame-level AUROC on the configured evaluation set.

## Pretrained weights

Available on request.

## Limitations

Training uses a single manipulation source, so performance degrades on unseen generators.

Heavy compression attenuates the bands the spectral filter operates on, which is visible in the FF++ C40 result.

The filter applies diagonal scaling rather than full complex multiplication and cannot adjust phase. A magnitude and phase parameterization with bounded gain remains untested.

The method predates diffusion-based face synthesis and has not been evaluated on diffusion-generated content.

No disaggregated evaluation across demographic groups was carried out. The benchmarks used lack annotations for skin tone, gender and age, so differences in error rates across groups are unmeasured, and faces with atypical appearance may be misclassified at higher rates.

## References

[1] J. Cao, C. Ma, T. Yao, S. Chen, S. Ding, X. Yang. End-to-End Reconstruction-Classification Learning for Face Forgery Detection. CVPR 2022. [code](https://github.com/VISION-SJTU/RECCE)

[2] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, M. Niessner. FaceForensics++: Learning to Detect Manipulated Facial Images. ICCV 2019.

[3] Y. Li, X. Yang, P. Sun, H. Qi, S. Lyu. Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics. CVPR 2020.

[4] B. Zi, M. Chang, J. Chen, X. Ma, Y. Jiang. WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection. ACM Multimedia 2020.

[5] B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang, C. Canton-Ferrer. The DeepFake Detection Challenge Dataset. arXiv:2006.07397, 2020.

[6] J. Deng, J. Guo, E. Ververas, I. Kotsia, S. Zafeiriou. RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild. CVPR 2020.

[7] Reported cross-dataset figures for RECCE are taken from the comparison table in arXiv:2411.05335.

## Acknowledgement

The reconstruction-classification backbone is based on the official RECCE implementation [1].

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

Sneha Kumari, sneha.k.1996@gmail.com
