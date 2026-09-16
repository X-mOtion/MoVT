# MoVT

## Video-Augmented Motion Tokenizer for Text-to-Motion Generation

**ACM MM 2026 · Oral**

**Inference code: available now.** Full training release: coming soon.

[Project page](https://x-motion.github.io/MoVT-website/) · [Paper](https://arxiv.org/abs/2609.14965)

MoVT enriches a 3D motion vocabulary with motion patterns learned from human action videos. It aligns 2D and 3D codebooks and uses shared token indices for text-conditioned motion generation.

### Release status

The inference pipeline is released in [`inference/`](inference/README.md). It
runs text-to-motion generation end to end: one English prompt produces an
aligned 2D motion, a 3D motion, and a side-by-side video of both.

Training code, the video-side codebook augmentation stage, and model
checkpoints are not included yet. Release updates will be posted here; no date
is announced for the full release.

The project page contains the abstract, teaser, pipeline, paper-reported results, and interactive motion examples.

### Acknowledgements

The released inference code builds on
[MoMask](https://github.com/EricGuo5513/momask-codes) for its RVQ-VAE
tokenizers, masked and residual transformers, and HumanML3D evaluation
protocol. Those components are used under the MIT licence; see
[`inference/LICENSE`](inference/LICENSE). Please also follow the licences of
HumanML3D, Motion-X++, MMPose and MoMask when using this code.

### Citation

A BibTeX entry will be added once the camera-ready version is available.
