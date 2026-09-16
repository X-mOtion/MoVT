"""Checkpoint names and sampling defaults for inference."""

from dataclasses import dataclass


#: 3D RVQ-VAE tokenizer (66D: 22 joints x 3 coordinates).
DEFAULT_VQ3D_NAME = "rvq_name"
DEFAULT_VQ3D_CKPT = "latest.tar"

#: 2D tokenizer, aligned to the 3D base codebook (44D: 22 joints x 2).
DEFAULT_VQ2D_NAME = "rvq_hml3d_xy_2d_supervised_align66_m4096e3_ce2"
DEFAULT_VQ2D_CKPT = "net_best_rec.tar"

#: Text-conditioned masked transformer (predicts the shared base tokens).
DEFAULT_MTRANS_NAME = "mtrans_rvq_name_66d_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns"
DEFAULT_MTRANS_CKPT = "net_best_loss.tar"

#: Residual transformer (predicts 3D RVQ layers 1..5).
DEFAULT_RTRANS_NAME = "rtrans_rvq_name_66d_nlayer8_nhead6_ld384_ff1024_cdp0.2_sw"
DEFAULT_RTRANS_CKPT = "net_best_loss.tar"


@dataclass(frozen=True)
class CheckpointConfig:
    """Where the four inference checkpoints live."""

    root: str = "checkpoints"
    dataset_name: str = "t2m"
    vq3d_name: str = DEFAULT_VQ3D_NAME
    vq3d_ckpt: str = DEFAULT_VQ3D_CKPT
    vq2d_name: str = DEFAULT_VQ2D_NAME
    vq2d_ckpt: str = DEFAULT_VQ2D_CKPT
    mtrans_name: str = DEFAULT_MTRANS_NAME
    mtrans_ckpt: str = DEFAULT_MTRANS_CKPT
    rtrans_name: str = DEFAULT_RTRANS_NAME
    rtrans_ckpt: str = DEFAULT_RTRANS_CKPT


@dataclass(frozen=True)
class SamplingConfig:
    """Decoding hyper-parameters.

    The defaults are the tuned values used for the reported HumanML3D numbers.
    """

    seed: int = 10107
    time_steps: int = 10
    cond_scale: float = 4.0
    temperature: float = 1.0
    topk: float = 0.9
    residual_cond_scale: float = 5.0
    residual_temperature: float = 1.0
    gumbel_sample: bool = False
