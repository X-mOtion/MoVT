"""Turning decoder output into joint coordinates."""

import numpy as np


def validate_motion_length(motion_length):
    """The tokenizer works on 4-frame patches within a bounded window."""

    if motion_length < 4 or motion_length > 196 or motion_length % 4:
        raise ValueError("motion_length must be a multiple of 4 in [4, 196]")
    return motion_length


def denormalize_joints(decoded, mean, std, joints_num, channels, frames):
    """Reshape ``(1, T, joints*channels)`` decoder output to ``(T, joints, channels)``.

    ``mean``/``std`` may be stored either per channel or per flattened feature;
    both layouts appear in the checkpoints handled here.
    """

    decoded = decoded.detach().cpu().numpy()[0, :frames]
    expected_width = joints_num * channels
    if decoded.ndim != 2 or decoded.shape[1] != expected_width:
        raise ValueError(
            f"decoder returned {decoded.shape}, expected (T, {expected_width})"
        )

    joints = decoded.reshape(decoded.shape[0], joints_num, channels)
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean.size == channels:
        mean = mean.reshape(1, 1, channels)
        std = std.reshape(1, 1, channels)
    elif mean.size == expected_width:
        mean = mean.reshape(1, joints_num, channels)
        std = std.reshape(1, joints_num, channels)
    else:
        raise ValueError(
            f"normalization stats contain {mean.size} values; expected "
            f"{channels} or {expected_width}"
        )
    return (joints * std + mean).astype(np.float32)
