# MoVT

## Video-Augmented Motion Tokenizer for Text-to-Motion Generation

**ACM MM 2026 · Oral**

**Inference code: available now.** Full training release: coming soon.

[Project page](https://x-motion.github.io/MoVT-website/) · [Paper](https://arxiv.org/abs/2609.14965)

MoVT enriches a 3D motion vocabulary with motion patterns learned from human action videos. It aligns 2D and 3D codebooks and uses shared token indices for text-conditioned motion generation.

The project page contains the abstract, teaser, pipeline, paper-reported results, and interactive motion examples.

### Release status

This repository currently holds an inference-only slice of MoVT: one English
prompt produces an aligned 2D motion, a 3D motion, and a side-by-side video of
both. Training code, the video-side codebook augmentation stage, datasets and
evaluation are not included yet. Release updates will be posted here; no date
is announced for the full release.

## Quick start

```bash
# 1. download and unpack the checkpoints (see the next section), then:
python infer_text_2d3d.py --text "a person walks forward and waves with their right hand"
python selfcheck.py          # arrays + video invariant check, <1 min on a GPU
```

That writes `motion_2d3d.mp4` into `outputs/text_2d3d/`, next to the `.npy`
outputs: the left panel is the 2D tokenizer output, the right panel is the 3D
motion, animated in lockstep. Pass `--no_video` to skip rendering.

`TextToMotion` loads all four networks once, so several prompts can be generated
without paying the load cost again:

```python
from movt import CheckpointConfig, TextToMotion, resolve_device
from movt.render import render_sample

model = TextToMotion(CheckpointConfig(), resolve_device(0))
sample = model.generate("a person walks forward and waves", motion_length=64)
sample.save("outputs/demo")
render_sample(sample, "outputs/demo/motion_2d3d.mp4")
print(sample.motion_3d.shape, sample.motion_2d.shape)
```

## Checkpoints

Inference needs four checkpoints, 774 MB in total. They are not stored in this
repository.

| checkpoint | role |
| --- | --- |
| `rvq_name` | 3D RVQ-VAE tokenizer, 66D, 6 quantizer layers |
| `rvq_hml3d_xy_2d_supervised_align66_m4096e3_ce2` | aligned 2D tokenizer, 44D, 1 layer |
| `mtrans_rvq_name_66d_..._rvq6ns` | masked transformer, predicts the shared base tokens |
| `rtrans_rvq_name_66d_..._sw` | residual transformer, fills RVQ layers 1-5 |

### Download

- **Baidu Netdisk** — [MoVT-inference-ckpt.zip](https://pan.baidu.com/s/16YORJ0nqQUXoTBCDnoxdgQ?pwd=uk6s)
  (extraction code: `uk6s`)

### Install

Unpack the archive at the repository root, so that `checkpoints/t2m/` appears
next to `movt/`:

```bash
unzip MoVT-inference-ckpt.zip        # creates ./checkpoints, ./SHA256SUMS.txt
cd checkpoints && sha256sum -c ../SHA256SUMS.txt && cd ..
python infer_text_2d3d.py --text "a person walks forward and waves"
```

`--checkpoints_dir` defaults to `checkpoints`, so nothing else needs setting.
The archive also contains a `README.md` repeating these steps and listing the
SHA256 of every file.

`prepare_assets.sh` is only for people who already have a MoVT training
checkout and would rather link the weights than copy them:

```bash
MOVT_SOURCE_ROOT=/path/to/MoVT-training-checkout ./prepare_assets.sh
```

## Outputs

From a single prompt, `outputs/text_2d3d/` gets:

| file | shape | meaning |
| --- | --- | --- |
| `base_indices.npy` | `(T/4,)` | shared base tokens, predicted from text |
| `rvq_indices_3d.npy` | `(T/4, 6)` | full 3D RVQ indices; column 0 equals `base_indices` |
| `motion_2d_joints.npy` | `(T, 22, 2)` | 2D motion decoded from `base_indices` |
| `motion_3d_base_joints.npy` | `(T, 22, 3)` | 3D motion from the base layer only |
| `motion_3d_joints.npy` | `(T, 22, 3)` | full 3D motion after residual layers |
| `motion_2d3d.mp4` | | side-by-side 2D + 3D animation (unless `--no_video`) |
| `report.json` | | shapes, index ranges and consistency checks |

`T` is `--motion_length` in frames at 20 FPS, and must be a multiple of 4 in
`[4, 196]`.

### A note on the two branches

The 2D and 3D outputs are decoded from the *same* base indices, but they are two
independent decodes, not a projection of one another. The 2D tokenizer is
genuinely two-dimensional — its decoder's output layer has 44 channels (22
joints x 2 axes) — and it was trained on the frontal `(x, y)` projection of the
motion. Measured agreement between the two decodes is 0.96 (x) and 0.998 (y) in
correlation, so they are close but not identical. If you need the exact frontal
projection of the 3D result, slice `motion_3d[..., :2]`.

## Video

`render_sample` draws both branches in one animated figure:

| panel | content |
| --- | --- |
| left | 2D tokenizer output, plotted as the frontal projection it is: `(x, y)` = (lateral, height) |
| right | 3D motion with true axis proportions, a ground plane, and a trail of the root path |

Both skeletons use the same five chain colours, so the same limb is the same
colour in both panels. Defaults: `20` fps, `110` dpi, a `12`-frame trail that
follows the root. Useful flags: `--no_video`, `--video_path`, `--video_fps`,
`--video_dpi`.

`movt/render.py` deliberately does not reuse MoMask's `utils/plot_script.py`.
That version calls `Axes3D(fig)` and sets `ax.dist`, both removed in newer
matplotlib, and on matplotlib >= 3.6 it silently saves a **single blank frame**
instead of failing. Here every artist is created once and updated in place,
which works on matplotlib 3.4 through 3.10 and is also faster.

## Pipeline

```text
text ──CLIP ViT-B/32──► masked transformer ──► base indices (1024-way codebook)
                                                     │
                     ┌───────────────────────────────┴──────────────────────────┐
                     ▼                                                          ▼
        2D tokenizer decoder                                       3D RVQ layer 0
        ──► (T, 22, 2)                                             residual transformer
                                                                   ──► layers 1..5
                                                                   ──► 3D tokenizer decoder
                                                                   ──► (T, 22, 3)
```

The masked transformer predicts the shared base tokens with ten rounds of
masked denoising. Those exact indices are then used twice: the 2D tokenizer
looks them up directly in its single-layer codebook, while the residual
transformer predicts the five remaining 3D RVQ layers layer-by-layer. The 3D
decoder sums the six code vectors and decodes them with one decoder.

The two codebooks are aligned in *index semantics*, not by sharing vectors:
token `k` of the 2D codebook and token `k` of the 3D base codebook represent the
same motion primitive, which is why the same indices can drive both branches.

## Layout

```text
infer_text_2d3d.py          entry point (argparse -> movt.cli)
selfcheck.py                runs one prompt and asserts the output invariants
prepare_assets.sh           links or copies checkpoints from a training checkout
requirements.txt

movt/
├── cli.py                  command line front end
├── config.py               checkpoint names + sampling defaults (dataclasses)
├── pipeline.py             TextToMotion: load once, generate many; MotionSample
├── postprocess.py          denormalisation and motion-length validation
├── kinematics.py           22-joint skeleton chains and the 2D projection axes
├── render.py               side-by-side 2D + 3D video
├── loaders/
│   ├── opt.py              reading opt.txt / meta/, picking checkpoints
│   ├── tokenizer.py        load_vq_model, load_stats
│   └── transformer.py      load_mask_transformer, load_res_transformer
├── models/                 network definitions
│   ├── vq/                 RVQ-VAE: encdec, residual VQ, quantizer, resnet
│   └── mask_transformer/   MaskTransformer + ResidualTransformer
└── utils/                  opt.txt parsing, POS table, seeding
```

## Environment

PyTorch + OpenAI CLIP + einops + matplotlib. CLIP `ViT-B/32` weights are
downloaded on first run and cached by the `clip` package.

Verified end to end on the local conda environments `NAR`, `comfyui`, `genmo`
and `livetalking-local`. In `mogen` (torch 2.5 + cuDNN) CLIP attention hits
`RuntimeError: cuDNN Frontend error: No valid engine configs`, an environment
issue rather than a code one; disabling that one fused kernel works around it:

```bash
python -c "
import torch, sys
torch.backends.cuda.enable_cudnn_sdp(False)
sys.argv = ['infer', '--text', 'a person walks forward', '--motion_length', '64']
from movt.cli import main; main()"
```

## Fidelity

`movt/models/mask_transformer/transformer.py` and `movt/models/vq/model.py` are
trimmed copies of MoMask's files. Removed: training losses, the text-editing
entry points (`edit`, `edit_beta`), the training-only `q_schedule` sampler, and
unused experimental helpers. No network definition, parameter, or inference
method was touched.

Verified: at the same seed this package reproduces MoMask-based full-repository
`infer_text_2d3d.py` output bit-for-bit across all five saved arrays.

## Acknowledgements

The network definitions under `movt/models/` are derived from
[MoMask](https://github.com/EricGuo5513/momask-codes) and are used under the MIT
licence reproduced in [`LICENSE`](LICENSE). HumanML3D supplies the motion
representation and text prompts this pipeline targets. Please also follow the
licences of HumanML3D, Motion-X++, MMPose and MoMask when using this code.

The released weights are trained on HumanML3D, which is licensed for
non-commercial research use.

## Citation

A BibTeX entry will be added once the camera-ready version is available.
