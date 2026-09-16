"""End-to-end check for the minimal text-to-motion path.

Runs one prompt through the full pipeline, including the video, and verifies
the invariants that define MoVT's aligned 2D/3D output. Takes well under a
minute on a GPU.

    python selfcheck.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MOTION_LENGTH = 8
TOKENS = MOTION_LENGTH // 4


def probe_frame_count(video_path):
    """Frame count via ffprobe, or ``None`` when ffprobe is unavailable."""

    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nw=1:nk=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def check_arrays(out):
    base = np.load(out / "base_indices.npy")
    rvq = np.load(out / "rvq_indices_3d.npy")
    m2d = np.load(out / "motion_2d_joints.npy")
    m3d = np.load(out / "motion_3d_joints.npy")

    assert base.shape == (TOKENS,), base.shape
    assert rvq.shape == (TOKENS, 6), rvq.shape
    assert m2d.shape == (MOTION_LENGTH, 22, 2), m2d.shape
    assert m3d.shape == (MOTION_LENGTH, 22, 3), m3d.shape
    assert np.array_equal(base, rvq[:, 0]), "layer 0 must equal the shared base indices"
    assert np.isfinite(m2d).all() and np.isfinite(m3d).all()
    assert base.min() >= 0 and base.max() < 1024

    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "OK" and report["base_matches_3d_layer0"] is True
    print("  arrays OK: shared base indices decode to aligned 2D and 3D motion")


def check_video(out):
    video = out / "motion_2d3d.mp4"
    if not video.exists():
        fallback = out / "motion_2d3d.gif"
        assert fallback.exists(), "no video was written"
        video = fallback
    assert video.stat().st_size > 5000, f"{video} looks empty"

    # The failure mode this guards against is an animation that silently
    # collapses to a single blank frame.
    frames = probe_frame_count(video)
    if frames is not None:
        assert frames == MOTION_LENGTH, f"expected {MOTION_LENGTH} frames, got {frames}"
    print(f"  video OK: {video.name}, {video.stat().st_size // 1024} KiB, frames={frames}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                sys.executable,
                str(HERE / "infer_text_2d3d.py"),
                "--text",
                "a person walks forward then stops",
                "--motion_length",
                str(MOTION_LENGTH),
                "--output_dir",
                tmp,
            ],
            cwd=HERE,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        out = Path(tmp)
        check_arrays(out)
        check_video(out)

    print("selfcheck OK")


if __name__ == "__main__":
    main()
