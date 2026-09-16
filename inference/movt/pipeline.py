"""Text -> shared base tokens -> aligned 2D and 3D motion."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import CheckpointConfig, SamplingConfig
from .loaders import (
    load_mask_transformer,
    load_res_transformer,
    load_stats,
    load_vq_model,
)
from .loaders.opt import resolve_checkpoint
from .postprocess import denormalize_joints, validate_motion_length
from .utils.fixseed import fixseed

JOINTS_NUM = 22
FPS = 20


def resolve_device(gpu_id):
    """``gpu_id >= 0`` selects CUDA when available, otherwise CPU."""

    if gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


@dataclass
class MotionSample:
    """Everything one prompt produces.

    ``motion_2d`` and ``motion_3d_base`` are decoded from the very same token
    sequence, which is what makes the pair aligned.
    """

    text: str
    base_indices: np.ndarray  # (T/4,)     shared 2D+3D base tokens
    rvq_indices: np.ndarray  # (T/4, 6)   full 3D RVQ indices
    motion_2d: np.ndarray  # (T, 22, 2)
    motion_3d_base: np.ndarray  # (T, 22, 3)  base layer only
    motion_3d: np.ndarray  # (T, 22, 3)  after residual layers
    seed: int = 0
    device: str = "cpu"
    codebook_size: int = 0
    code_dim: int = 0

    @property
    def motion_length(self):
        return int(self.motion_3d.shape[0])

    @property
    def base_matches_3d_layer0(self):
        return bool(np.array_equal(self.base_indices, self.rvq_indices[:, 0]))

    def report(self):
        return {
            "status": "OK",
            "text": self.text,
            "seed": self.seed,
            "device": self.device,
            "motion_length": self.motion_length,
            "fps": FPS,
            "token_count": int(self.base_indices.shape[0]),
            "shared_codebook_size": self.codebook_size,
            "code_dimension": self.code_dim,
            "three_d_quantizer_layers": int(self.rvq_indices.shape[1]),
            "two_d_quantizer_layers": 1,
            "base_index_range": [
                int(self.base_indices.min()),
                int(self.base_indices.max()),
            ],
            "base_matches_3d_layer0": self.base_matches_3d_layer0,
            "motion_2d_shape": list(self.motion_2d.shape),
            "motion_3d_base_shape": list(self.motion_3d_base.shape),
            "motion_3d_shape": list(self.motion_3d.shape),
            "finite_outputs": bool(
                np.isfinite(self.motion_2d).all()
                and np.isfinite(self.motion_3d_base).all()
                and np.isfinite(self.motion_3d).all()
            ),
            "index_semantics": (
                "The 2D indices and 3D RVQ layer-0 indices are identical; "
                "3D RVQ layers 1-5 are additional residual-detail indices."
            ),
        }

    def save(self, output_dir):
        """Write the index and joint arrays plus ``report.json``."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "base_indices.npy", self.base_indices)
        np.save(output_dir / "rvq_indices_3d.npy", self.rvq_indices)
        np.save(output_dir / "motion_2d_joints.npy", self.motion_2d)
        np.save(output_dir / "motion_3d_base_joints.npy", self.motion_3d_base)
        np.save(output_dir / "motion_3d_joints.npy", self.motion_3d)

        report = self.report()
        with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return output_dir


class TextToMotion:
    """Loaded, reusable text-to-motion model.

    Construction loads all four networks; ``generate`` can then be called many
    times without paying that cost again.
    """

    def __init__(self, checkpoints, device=None):
        self.checkpoints = checkpoints
        self.device = device or resolve_device(-1)

        root = Path(checkpoints.root) / checkpoints.dataset_name
        self.vq3d_dir = root / checkpoints.vq3d_name
        self.vq2d_dir = root / checkpoints.vq2d_name

        self.vq3d, self.vq3d_opt, dim3d = load_vq_model(
            str(self.vq3d_dir),
            self.device,
            str(resolve_checkpoint(self.vq3d_dir, checkpoints.vq3d_ckpt)),
        )
        self.vq2d, self.vq2d_opt, dim2d = load_vq_model(
            str(self.vq2d_dir),
            self.device,
            str(resolve_checkpoint(self.vq2d_dir, checkpoints.vq2d_ckpt)),
        )
        self._validate_tokenizers(dim3d, dim2d)

        self.mask_transformer = load_mask_transformer(
            checkpoints.root,
            checkpoints.dataset_name,
            checkpoints.mtrans_name,
            checkpoints.mtrans_ckpt,
            self.vq3d_opt,
            self.device,
        ).to(self.device).eval()
        self.residual_transformer = load_res_transformer(
            checkpoints.root,
            checkpoints.dataset_name,
            checkpoints.rtrans_name,
            checkpoints.rtrans_ckpt,
            self.vq3d_opt,
            self.device,
        ).to(self.device).eval()

        self.mean_3d, self.std_3d = load_stats(str(self.vq3d_dir))
        self.mean_2d, self.std_2d = load_stats(str(self.vq2d_dir))

    def _validate_tokenizers(self, dim3d, dim2d):
        if dim3d != 66 or dim2d != 44:
            raise ValueError(f"expected 66D/44D tokenizers, got {dim3d}D/{dim2d}D")
        if int(self.vq3d_opt.nb_code) != int(self.vq2d_opt.nb_code):
            raise ValueError("2D and 3D base codebooks have different sizes")
        if int(self.vq3d_opt.code_dim) != int(self.vq2d_opt.code_dim):
            raise ValueError("2D and 3D code vectors have different dimensions")
        if int(self.vq2d_opt.num_quantizers) != 1:
            raise ValueError("the aligned 2D tokenizer must contain one quantizer")

    @classmethod
    def from_gpu(cls, checkpoints=None, gpu_id=0):
        return cls(checkpoints or CheckpointConfig(), resolve_device(gpu_id))

    # -- generation ------------------------------------------------------- #

    def generate(self, text, motion_length=64, sampling=None):
        """Run the full pipeline for one English prompt."""

        sampling = sampling or SamplingConfig()
        validate_motion_length(motion_length)
        fixseed(sampling.seed)

        token_lens = torch.tensor(
            [motion_length // 4], device=self.device, dtype=torch.long
        )
        prompts = [text]

        with torch.inference_mode():
            base_ids = self.mask_transformer.generate(
                prompts,
                token_lens,
                sampling.time_steps,
                sampling.cond_scale,
                temperature=sampling.temperature,
                topk_filter_thres=sampling.topk,
                gsample=sampling.gumbel_sample,
                force_mask=False,
            )
            rvq_ids = self.residual_transformer.generate(
                base_ids,
                prompts,
                token_lens,
                temperature=sampling.residual_temperature,
                topk_filter_thres=sampling.topk,
                cond_scale=sampling.residual_cond_scale,
            )
            decoded_2d = self.vq2d.forward_decoder_firstlayer(base_ids)
            decoded_3d_base = self.vq3d.forward_decoder_firstlayer(base_ids)
            decoded_3d = self.vq3d.forward_decoder(rvq_ids)

        sample = MotionSample(
            text=text,
            base_indices=base_ids.detach().cpu().numpy()[0].astype(np.int64),
            rvq_indices=rvq_ids.detach().cpu().numpy()[0].astype(np.int64),
            motion_2d=denormalize_joints(
                decoded_2d, self.mean_2d, self.std_2d, JOINTS_NUM, 2, motion_length
            ),
            motion_3d_base=denormalize_joints(
                decoded_3d_base, self.mean_3d, self.std_3d, JOINTS_NUM, 3, motion_length
            ),
            motion_3d=denormalize_joints(
                decoded_3d, self.mean_3d, self.std_3d, JOINTS_NUM, 3, motion_length
            ),
            seed=sampling.seed,
            device=str(self.device),
            codebook_size=int(self.vq3d_opt.nb_code),
            code_dim=int(self.vq3d_opt.code_dim),
        )
        self._check(sample)
        return sample

    def _check(self, sample):
        """Fail loudly instead of writing silently wrong motion."""

        if not sample.base_matches_3d_layer0:
            raise RuntimeError("3D RVQ layer 0 no longer matches the shared base indices")
        if sample.base_indices.min() < 0 or sample.base_indices.max() >= sample.codebook_size:
            raise RuntimeError("generated base index is outside the shared codebook")
        if not sample.report()["finite_outputs"]:
            raise RuntimeError("a decoded motion contains NaN or infinity")
