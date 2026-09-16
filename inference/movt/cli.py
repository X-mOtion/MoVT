"""Command line front end for text-to-motion inference."""

import argparse
import json
from pathlib import Path

from .config import (
    DEFAULT_MTRANS_CKPT,
    DEFAULT_MTRANS_NAME,
    DEFAULT_RTRANS_CKPT,
    DEFAULT_RTRANS_NAME,
    DEFAULT_VQ2D_CKPT,
    DEFAULT_VQ2D_NAME,
    DEFAULT_VQ3D_CKPT,
    DEFAULT_VQ3D_NAME,
    CheckpointConfig,
    SamplingConfig,
)
from .pipeline import TextToMotion, resolve_device


def build_parser():
    parser = argparse.ArgumentParser(
        description="Text -> shared base indices -> aligned 2D and 3D motions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--text", required=True, help="English motion description")
    parser.add_argument("--motion_length", type=int, default=64, help="frames at 20 FPS")
    parser.add_argument("--seed", type=int, default=SamplingConfig.seed)
    parser.add_argument("--gpu_id", type=int, default=0, help="-1 selects CPU")
    parser.add_argument("--output_dir", default="outputs/text_2d3d")

    group = parser.add_argument_group("checkpoints")
    group.add_argument("--checkpoints_dir", default=CheckpointConfig.root)
    group.add_argument("--dataset_name", default=CheckpointConfig.dataset_name)
    group.add_argument("--vq3d_name", default=DEFAULT_VQ3D_NAME)
    group.add_argument("--vq2d_name", default=DEFAULT_VQ2D_NAME)
    group.add_argument("--vq3d_ckpt", default=DEFAULT_VQ3D_CKPT)
    group.add_argument("--vq2d_ckpt", default=DEFAULT_VQ2D_CKPT)
    group.add_argument("--mtrans_name", default=DEFAULT_MTRANS_NAME)
    group.add_argument("--rtrans_name", default=DEFAULT_RTRANS_NAME)
    group.add_argument("--mtrans_ckpt", default=DEFAULT_MTRANS_CKPT)
    group.add_argument("--rtrans_ckpt", default=DEFAULT_RTRANS_CKPT)

    group = parser.add_argument_group("sampling")
    group.add_argument("--time_steps", type=int, default=SamplingConfig.time_steps)
    group.add_argument("--cond_scale", type=float, default=SamplingConfig.cond_scale)
    group.add_argument("--temperature", type=float, default=SamplingConfig.temperature)
    group.add_argument("--topkr", type=float, default=SamplingConfig.topk)
    group.add_argument(
        "--res_cond_scale", type=float, default=SamplingConfig.residual_cond_scale
    )
    group.add_argument(
        "--res_temperature", type=float, default=SamplingConfig.residual_temperature
    )
    group.add_argument("--gumbel_sample", action="store_true")

    group = parser.add_argument_group("video")
    group.add_argument(
        "--no_video",
        action="store_true",
        help="skip the side-by-side 2D/3D video",
    )
    group.add_argument(
        "--video_path",
        default=None,
        help="defaults to <output_dir>/motion_2d3d.mp4",
    )
    group.add_argument("--video_fps", type=int, default=20)
    group.add_argument("--video_dpi", type=int, default=110)
    return parser


def configs_from_args(args):
    """Split the flat CLI namespace into the two config objects."""

    checkpoints = CheckpointConfig(
        root=args.checkpoints_dir,
        dataset_name=args.dataset_name,
        vq3d_name=args.vq3d_name,
        vq3d_ckpt=args.vq3d_ckpt,
        vq2d_name=args.vq2d_name,
        vq2d_ckpt=args.vq2d_ckpt,
        mtrans_name=args.mtrans_name,
        mtrans_ckpt=args.mtrans_ckpt,
        rtrans_name=args.rtrans_name,
        rtrans_ckpt=args.rtrans_ckpt,
    )
    sampling = SamplingConfig(
        seed=args.seed,
        time_steps=args.time_steps,
        cond_scale=args.cond_scale,
        temperature=args.temperature,
        topk=args.topkr,
        residual_cond_scale=args.res_cond_scale,
        residual_temperature=args.res_temperature,
        gumbel_sample=args.gumbel_sample,
    )
    return checkpoints, sampling


def main(argv=None):
    args = build_parser().parse_args(argv)
    checkpoints, sampling = configs_from_args(args)

    model = TextToMotion(checkpoints, resolve_device(args.gpu_id))
    sample = model.generate(args.text, args.motion_length, sampling)
    output_dir = sample.save(args.output_dir)

    video_path = None
    if not args.no_video:
        from .render import render_sample  # matplotlib is only needed here

        target = args.video_path or (Path(args.output_dir) / "motion_2d3d.mp4")
        video_path = render_sample(
            sample, target, fps=args.video_fps, dpi=args.video_dpi
        )

    print(json.dumps(sample.report(), ensure_ascii=False, indent=2))
    print(f"Saved aligned inference artifacts to {Path(output_dir).resolve()}")
    if video_path is not None:
        print(f"Saved 2D+3D video to {Path(video_path).resolve()}")
    return sample
