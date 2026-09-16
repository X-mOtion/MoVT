"""Render a :class:`~movt.pipeline.MotionSample` as a 2D + 3D video.

Both views share one frame: the left panel shows the 2D tokenizer output, the
right panel shows the 3D motion it is decoded alongside. Because the two are
decoded from the same base tokens, the video is a direct visual check that the
alignment holds.

The implementation deliberately avoids the partner repository's
``utils/plot_script.py``: that version relies on ``Axes3D(fig)`` and
``ax.dist``, which newer matplotlib releases removed, so it silently emits a
single blank frame. Here every artist is created once and updated in place,
which is also faster.
"""

import shutil
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from .kinematics import (  # noqa: E402
    AXIS_X,
    AXIS_Y,
    AXIS_Z,
    CHAIN_COLORS,
    KINEMATIC_CHAIN,
)

DEFAULT_FPS = 20
DEFAULT_DPI = 110

#: The trajectory trail is drawn just above the ground so it is not z-fought
#: by the ground plane.
TRAIL_LIFT = 0.02


def _center_on_root(joints, horizontal_axes):
    """Shift every frame so the root sits at the origin horizontally."""

    out = np.array(joints, dtype=np.float64)
    for axis in horizontal_axes:
        out[..., axis] -= out[:, :1, axis]
    return out


def _drop_to_ground(joints, up_axis):
    """Put the lowest point of the sequence on the ground plane."""

    out = joints.copy()
    out[..., up_axis] -= out[..., up_axis].min()
    return out


def _limits(values, pad_ratio=0.15, min_span=1.0):
    values = np.asarray(values)[np.isfinite(values)]
    lo, hi = float(np.min(values)), float(np.max(values))
    span = max(hi - lo, min_span)
    pad = span * pad_ratio
    return lo - pad, hi + pad


def _symmetric(lo, hi, min_half):
    """Widen a range so it is centred on zero, keeping the character centred."""

    half = max(abs(lo), abs(hi), min_half)
    return -half, half


def _trail_offsets(root_path, window):
    """Root positions relative to the current frame, over a sliding window.

    Returns an array of shape ``(T, window, dim)`` padded with NaN so the trail
    can be drawn as a single line per frame.
    """

    frames = root_path.shape[0]
    dim = root_path.shape[1]
    trail = np.full((frames, window, dim), np.nan)
    for index in range(frames):
        start = max(0, index - window + 1)
        segment = root_path[start : index + 1] - root_path[index]
        trail[index, : segment.shape[0]] = segment
    return trail


def render_sample(sample, output_path, fps=DEFAULT_FPS, dpi=DEFAULT_DPI, trail_window=12):
    """Write a side-by-side 2D/3D video for one motion sample.

    ``output_path`` should end in ``.mp4``; if ffmpeg is unavailable the frames
    are written next to it as a ``.gif`` instead, and the realised path is
    returned.
    """

    # The trail has to come from the *world* trajectory: once the motion is
    # centred on the root, the root position is identically zero.
    raw_3d = np.asarray(sample.motion_3d, dtype=np.float64)
    root_path = raw_3d[:, 0][:, [AXIS_X, AXIS_Z]]
    trail = _trail_offsets(root_path, trail_window)

    motion_3d = _drop_to_ground(
        _center_on_root(sample.motion_3d, (AXIS_X, AXIS_Z)), AXIS_Y
    )
    motion_2d = _drop_to_ground(_center_on_root(sample.motion_2d, (AXIS_X,)), AXIS_Y)

    frames = motion_3d.shape[0]
    output_path = Path(output_path)

    chains = list(zip(KINEMATIC_CHAIN, CHAIN_COLORS))

    # Axis ranges: the body plus whatever the visible trail spans.
    x_lo, x_hi = _limits(np.concatenate([motion_3d[..., AXIS_X].ravel(), trail[..., 0].ravel()]))
    z_lo, z_hi = _limits(np.concatenate([motion_3d[..., AXIS_Z].ravel(), trail[..., 1].ravel()]))
    y_lo, y_hi = _limits(motion_3d[..., AXIS_Y].ravel(), min_span=1.6)
    x2_lo, x2_hi = _limits(motion_2d[..., AXIS_X].ravel())
    y2_lo, y2_hi = _limits(motion_2d[..., AXIS_Y].ravel(), min_span=1.6)

    height_span = max(y_hi - y_lo, y2_hi - y2_lo)
    x_lo, x_hi = _symmetric(x_lo, x_hi, 0.32 * height_span)
    z_lo, z_hi = _symmetric(z_lo, z_hi, 0.32 * height_span)
    x2_lo, x2_hi = _symmetric(x2_lo, x2_hi, 0.45 * height_span)

    figure = plt.figure(figsize=(12.6, 6.4), constrained_layout=True)
    figure.suptitle(sample.text, fontsize=13)

    ax2d = figure.add_subplot(1, 2, 1)
    ax3d = figure.add_subplot(1, 2, 2, projection="3d")

    # ---- 2D panel -------------------------------------------------------- #
    ax2d.set_title("2D tokenizer output (x, y)", fontsize=11)
    ax2d.set_xlim(x2_lo, x2_hi)
    ax2d.set_ylim(y2_lo, y2_hi)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.axhline(0.0, color="0.6", linewidth=1.0, zorder=0)
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y (up)")
    ax2d.grid(True, color="0.92", linewidth=0.6)
    ax2d.tick_params(labelsize=8)

    lines_2d = [
        ax2d.plot([], [], color=color, linewidth=3.0, solid_capstyle="round")[0]
        for _, color in chains
    ]

    # ---- 3D panel -------------------------------------------------------- #
    ax3d.set_title("3D motion (x, y, z)", fontsize=11)
    ax3d.set_xlim(x_lo, x_hi)
    ax3d.set_ylim(z_lo, z_hi)
    ax3d.set_zlim(y_lo, y_hi)
    ax3d.set_xlabel("x", fontsize=9)
    ax3d.set_ylabel("z (forward)", fontsize=9)
    ax3d.set_zlabel("y (up)", fontsize=9)
    ax3d.tick_params(labelsize=7)
    ax3d.view_init(elev=14, azim=-58)
    if hasattr(ax3d, "computed_zorder"):
        # Draw in insertion order instead of depth-sorting, so the ground plane
        # cannot end up painted over the trail that rests on it.
        ax3d.computed_zorder = False
    try:
        # True proportions, so the 3D view is not stretched.
        ax3d.set_box_aspect((x_hi - x_lo, z_hi - z_lo, y_hi - y_lo))
    except AttributeError:  # pragma: no cover - matplotlib < 3.3
        pass

    ground = Poly3DCollection(
        [
            [
                (x_lo, z_lo, 0.0),
                (x_lo, z_hi, 0.0),
                (x_hi, z_hi, 0.0),
                (x_hi, z_lo, 0.0),
            ]
        ],
        facecolor=(0.55, 0.55, 0.55, 0.35),
        edgecolor="0.7",
        linewidth=0.6,
    )
    ax3d.add_collection3d(ground)
    ground.set_zorder(1)
    (trail_line,) = ax3d.plot(
        [], [], [], color="#1f77b4", linewidth=2.2, alpha=0.95, zorder=5
    )

    lines_3d = [
        ax3d.plot(
            [], [], [], color=color, linewidth=3.0, solid_capstyle="round", zorder=10
        )[0]
        for _, color in chains
    ]

    def update(index):
        frame_2d = motion_2d[index]
        for line, (chain, _) in zip(lines_2d, chains):
            line.set_data(frame_2d[chain, AXIS_X], frame_2d[chain, AXIS_Y])

        frame_3d = motion_3d[index]
        for line, (chain, _) in zip(lines_3d, chains):
            line.set_data_3d(
                frame_3d[chain, AXIS_X],
                frame_3d[chain, AXIS_Z],
                frame_3d[chain, AXIS_Y],
            )

        visible = trail[index]
        mask = np.isfinite(visible[:, 0])
        trail_line.set_data_3d(
            visible[mask, 0], visible[mask, 1], np.full(int(mask.sum()), TRAIL_LIFT)
        )
        return lines_2d + lines_3d + [trail_line]

    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000.0 / fps, blit=False, repeat=False
    )

    if FFMpegWriter.isAvailable() and output_path.suffix.lower() == ".mp4":
        writer = FFMpegWriter(fps=fps, bitrate=3000)
        realised = output_path
    else:
        if output_path.suffix.lower() == ".mp4" and shutil.which("ffmpeg") is None:
            print("ffmpeg not found on PATH; writing a GIF instead")
        writer = PillowWriter(fps=fps)
        realised = output_path.with_suffix(".gif")

    realised.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(realised), writer=writer, dpi=dpi)
    plt.close(figure)
    return realised
