"""Reading the metadata stored next to each checkpoint.

Every checkpoint directory produced by this project keeps an ``opt.txt`` and a
``meta/`` folder. Those two are enough to rebuild a network without touching
the original training configuration.
"""

import os
from argparse import Namespace
from pathlib import Path
from os.path import join as pjoin

import numpy as np


def read_opt_file(opt_path):
    """Parse a plain ``opt.txt`` into a namespace."""

    opt = Namespace()
    with open(opt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("---") or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            if value in {"True", "False"}:
                value = value == "True"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            setattr(opt, key, value)

    defaults = {
        "code_dim": 512,
        "nb_code": 512,
        "mu": 0.99,
        "down_t": 2,
        "stride_t": 2,
        "width": 512,
        "depth": 3,
        "dilation_growth_rate": 3,
        "output_emb_width": 512,
        "vq_act": "relu",
        "vq_norm": None,
        "num_quantizers": 6,
        "shared_codebook": False,
        "quantize_dropout_prob": 0.2,
        "joints_num": 22,
    }
    for key, value in defaults.items():
        if not hasattr(opt, key):
            setattr(opt, key, value)
    return opt


def infer_dim_pose(vq_dir, opt):
    """Recover the tokenizer input width from the stored normalisation stats.

    This yields 66 for the 3D tokenizer and 44 for the 2D one.
    """

    if hasattr(opt, "dim_pose"):
        return int(opt.dim_pose)

    mean_path = pjoin(vq_dir, "meta", "mean.npy")
    if not os.path.exists(mean_path):
        raise FileNotFoundError(f"missing mean stats: {mean_path}")
    mean = np.load(mean_path)
    if mean.ndim == 1:
        return int(mean.shape[0])
    if mean.shape[1] == 1:
        return int(getattr(opt, "joints_num", 22) * mean.shape[-1])
    return int(np.prod(mean.shape[1:]))


def resolve_checkpoint(vq_dir, checkpoint):
    """Accept either a bare filename (resolved inside ``model/``) or a path."""

    path = Path(checkpoint)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return Path(vq_dir) / "model" / checkpoint


def choose_checkpoint(vq_dir, explicit=None):
    """Pick a checkpoint, preferring the best-FID one when none is given."""

    if explicit is not None:
        return explicit
    candidates = [
        "net_best_fid.tar",
        "net_best_rec.tar",
        "best_rec.tar",
        "latest.tar",
    ]
    for name in candidates:
        path = pjoin(vq_dir, "model", name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no VQ checkpoint found under {pjoin(vq_dir, 'model')}")


def checkpoint_state(ckpt_path):
    """Extract the model state dict from either checkpoint layout."""

    import torch  # imported lazily so opt parsing stays lightweight

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("vq_model", ckpt.get("net"))
    if state is None:
        raise KeyError(f"checkpoint has neither 'vq_model' nor 'net': {ckpt_path}")
    return state


def sync_vq_opt_from_state(opt, state):
    """Trust the stored codebooks over ``opt.txt`` for codebook geometry."""

    codebook_keys = sorted(
        key
        for key in state.keys()
        if key.startswith("quantizer.layers.") and key.endswith(".codebook")
    )
    if not codebook_keys:
        return opt

    first_codebook = state[codebook_keys[0]]
    opt.nb_code = int(first_codebook.shape[0])
    opt.code_dim = int(first_codebook.shape[1])
    opt.num_quantizers = len(codebook_keys)
    return opt
