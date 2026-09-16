"""Loading the masked and residual transformers."""

from os.path import join as pjoin

import torch

from ..models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from ..utils.get_opt import get_opt

CLIP_VERSION = "ViT-B/32"
CLIP_DIM = 512


def _transformer_dir(checkpoints_dir, dataset_name, name):
    return pjoin(checkpoints_dir, dataset_name, name)


def _load_state(path, keys, device):
    ckpt = torch.load(path, map_location=device)
    for key in keys:
        if key in ckpt:
            return ckpt[key], ckpt
    raise KeyError(f"none of {keys} found in {path}")


def load_mask_transformer(
    checkpoints_dir, dataset_name, name, ckpt_name, vq_opt, device
):
    """Load the masked transformer that predicts the shared base indices."""

    model_dir = _transformer_dir(checkpoints_dir, dataset_name, name)
    model_opt = get_opt(pjoin(model_dir, "opt.txt"), device)
    model_opt.num_tokens = int(vq_opt.nb_code)
    model_opt.num_quantizers = int(vq_opt.num_quantizers)
    model_opt.code_dim = int(vq_opt.code_dim)

    trans = MaskTransformer(
        code_dim=vq_opt.code_dim,
        cond_mode="text",
        latent_dim=model_opt.latent_dim,
        ff_size=model_opt.ff_size,
        num_layers=model_opt.n_layers,
        num_heads=model_opt.n_heads,
        dropout=model_opt.dropout,
        clip_dim=CLIP_DIM,
        cond_drop_prob=model_opt.cond_drop_prob,
        clip_version=CLIP_VERSION,
        opt=model_opt,
    )

    path = pjoin(model_dir, "model", ckpt_name)
    state, ckpt = _load_state(path, ("t2m_transformer", "trans"), device)
    missing, unexpected = trans.load_state_dict(state, strict=False)
    assert len(unexpected) == 0, unexpected
    assert all(k.startswith("clip_model.") for k in missing), missing
    print(f"Loaded Mask Transformer: {path}, ep={ckpt.get('ep')}")
    return trans


def load_res_transformer(
    checkpoints_dir, dataset_name, name, ckpt_name, vq_opt, device
):
    """Load the residual transformer that fills RVQ layers 1..N-1."""

    model_dir = _transformer_dir(checkpoints_dir, dataset_name, name)
    res_opt = get_opt(pjoin(model_dir, "opt.txt"), device)
    res_opt.num_tokens = int(vq_opt.nb_code)
    res_opt.num_quantizers = int(vq_opt.num_quantizers)
    res_opt.code_dim = int(vq_opt.code_dim)

    trans = ResidualTransformer(
        code_dim=vq_opt.code_dim,
        cond_mode="text",
        latent_dim=res_opt.latent_dim,
        ff_size=res_opt.ff_size,
        num_layers=res_opt.n_layers,
        num_heads=res_opt.n_heads,
        dropout=res_opt.dropout,
        clip_dim=CLIP_DIM,
        shared_codebook=vq_opt.shared_codebook,
        cond_drop_prob=res_opt.cond_drop_prob,
        share_weight=res_opt.share_weight,
        clip_version=CLIP_VERSION,
        opt=res_opt,
    )

    path = pjoin(model_dir, "model", ckpt_name)
    state, ckpt = _load_state(path, ("res_transformer",), device)
    missing, unexpected = trans.load_state_dict(state, strict=False)
    assert len(unexpected) == 0, unexpected
    assert all(k.startswith("clip_model.") for k in missing), missing
    print(f"Loaded Residual Transformer: {path}, ep={ckpt.get('ep')}")
    return trans
