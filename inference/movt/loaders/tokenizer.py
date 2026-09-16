"""Loading and using the RVQ-VAE motion tokenizers."""

from os.path import join as pjoin

import numpy as np
import torch

from ..models.vq.model import RVQVAE
from .opt import (
    choose_checkpoint,
    checkpoint_state,
    infer_dim_pose,
    read_opt_file,
    sync_vq_opt_from_state,
)


def load_vq_model(vq_dir, device, ckpt_path=None):
    """Load one RVQ-VAE tokenizer.

    Returns ``(model, opt, dim_pose)``; ``opt`` carries the codebook geometry
    that the transformers need in order to be built consistently.
    """

    opt = read_opt_file(pjoin(vq_dir, "opt.txt"))
    dim_pose = infer_dim_pose(vq_dir, opt)
    ckpt_path = choose_checkpoint(vq_dir, ckpt_path)
    state = checkpoint_state(ckpt_path)
    opt = sync_vq_opt_from_state(opt, state)

    model = RVQVAE(
        opt,
        dim_pose,
        opt.nb_code,
        opt.code_dim,
        opt.output_emb_width,
        opt.down_t,
        opt.stride_t,
        opt.width,
        opt.depth,
        opt.dilation_growth_rate,
        opt.vq_act,
        opt.vq_norm,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print(f"Loaded VQ model from {ckpt_path} with dim_pose={dim_pose}")
    return model, opt, dim_pose


def load_stats(vq_dir):
    """Load the per-channel normalisation statistics of a tokenizer."""

    mean = np.load(pjoin(vq_dir, "meta", "mean.npy")).astype(np.float32)
    std = np.load(pjoin(vq_dir, "meta", "std.npy")).astype(np.float32)
    if mean.ndim == 2:
        mean = mean.reshape(1, 1, mean.shape[-1])
        std = std.reshape(1, 1, std.shape[-1])
    return mean, std
