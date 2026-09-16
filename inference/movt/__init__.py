"""MoVT minimal text-to-motion inference.

One English prompt in, an aligned 2D and 3D motion out. The package is split
by responsibility:

``config``      checkpoint names and sampling defaults
``loaders``     rebuilding the four inference-time networks from checkpoints
``pipeline``    text -> base tokens -> aligned 2D/3D motion
``postprocess`` decoding and denormalisation helpers
``cli``         command line front end
``models``      network definitions (RVQ-VAE, masked/residual transformer)
``utils``       option-file parsing and seeding
"""

from .config import CheckpointConfig, SamplingConfig
from .pipeline import MotionSample, TextToMotion, resolve_device

__all__ = [
    "CheckpointConfig",
    "SamplingConfig",
    "MotionSample",
    "TextToMotion",
    "resolve_device",
]
