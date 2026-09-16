# MoVT minimal text-to-motion inference

Inference-only slice of MoVT: one English prompt in, an aligned 2D and 3D
motion out, plus a video showing both in the same frame. Training, datasets,
evaluation and the website are all left behind. Everything here is reachable
from `infer_text_2d3d.py`.

## Quick start

```bash
./prepare_assets.sh          # link the 4 needed checkpoints from ../checkpoints
python infer_text_2d3d.py --text "a person walks forward and waves with their right hand"
python selfcheck.py          # arrays + video invariant check, <1 min on a GPU
```

That writes `motion_2d3d.mp4` next to the `.npy` outputs: the left panel is the
2D tokenizer output, the right panel is the 3D motion, animated in lockstep.
Pass `--no_video` to skip rendering.

## Layout

```text
infer_text_2d3d.py          entry point (argparse -> movt.cli)
selfcheck.py                runs one prompt and asserts the output invariants
prepare_assets.sh           links or copies the 4 checkpoint directories
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

`TextToMotion` loads all four networks once, so several prompts can be
generated without paying the load cost again:

```python
from movt import CheckpointConfig, SamplingConfig, TextToMotion, resolve_device

model = TextToMotion(CheckpointConfig(), resolve_device(0))
sample = model.generate("a person walks forward and waves", motion_length=64)
sample.save("outputs/demo")
print(sample.motion_3d.shape, sample.motion_2d.shape)

from movt.render import render_sample
render_sample(sample, "outputs/demo/motion_2d3d.mp4")
```

## Outputs

`MotionSample.save()` writes to `outputs/text_2d3d/`:

| file | shape | meaning |
| --- | --- | --- |
| `base_indices.npy` | `(T/4,)` | shared base tokens, predicted from text |
| `rvq_indices_3d.npy` | `(T/4, 6)` | full 3D RVQ indices; column 0 equals `base_indices` |
| `motion_2d_joints.npy` | `(T, 22, 2)` | 2D projection decoded from `base_indices` |
| `motion_3d_base_joints.npy` | `(T, 22, 3)` | 3D motion from the base layer only |
| `motion_3d_joints.npy` | `(T, 22, 3)` | full 3D motion after residual layers |
| `motion_2d3d.mp4` | | side-by-side 2D + 3D animation (unless `--no_video`) |
| `report.json` | | shapes, index ranges and consistency checks |

`T` is `--motion_length` in frames at 20 FPS, and must be a multiple of 4 in
`[4, 196]`.

## Video

`render_sample` draws both branches of the pipeline in one animated figure:

| panel | content |
| --- | --- |
| left | 2D tokenizer output, plotted as the frontal projection it actually is: `(x, y)` = (lateral, height) |
| right | 3D motion with true axis proportions, a ground plane, and a trail of the root path |

Both skeletons are drawn with the same five chain colours, so the same limb is
the same colour in both panels. Rendering defaults: `20` fps, `110` dpi, a
`12`-frame trail that follows the root.

Useful flags: `--no_video`, `--video_path`, `--video_fps`, `--video_dpi`.

`movt/render.py` does not reuse the partner repository's `utils/plot_script.py`.
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

The 2D and 3D branches share the *same* base codebook, which is what makes the
pair aligned. Only layer 0 is shared; 3D layers 1-5 are extra residual detail
with no 2D counterpart.

## Environment

PyTorch + OpenAI CLIP + einops. CLIP `ViT-B/32` weights are downloaded on first
run and cached by the `clip` package.

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
trimmed copies of the parent repository's files. Removed: training losses, the
text-editing entry points (`edit`, `edit_beta`), the training-only `q_schedule`
sampler, and unused experimental helpers. No network definition, parameter, or
inference method was touched.

Verified: at the same seed this package reproduces the full-repository
`infer_text_2d3d.py` output bit-for-bit across all five saved arrays.

## Acknowledgements

The network definitions under `movt/models/` are derived from
[MoMask](https://github.com/EricGuo5513/momask-codes) and are used under the
MIT licence reproduced in [`LICENSE`](LICENSE). HumanML3D supplies the motion
representation and text prompts this pipeline targets.
