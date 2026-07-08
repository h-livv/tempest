"""
Build rollout comparison videos from pre-rendered frame images or legacy rollout NPZ files.

Usage:
    python ml/animate.py <rollout_frames_dir> [--output path.mp4] [--fps 30]
    python ml/animate.py <ml_run_dir>   # legacy: rollout_*.npz inside run folder
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

RUNS_BASE_DIR = os.path.join("ml", "runs")


def _build_video_from_frames(frames_dir: str, output_path: str, fps: int = 30) -> None:
    """Stitch sorted frame_*.png images into an MP4 using ffmpeg."""
    pattern = os.path.join(frames_dir, "frame_%04d.png")
    first_frame = os.path.join(frames_dir, "frame_0000.png")
    if not os.path.isfile(first_frame):
        frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        if not frames:
            raise FileNotFoundError(f"No frame_*.png images found in {frames_dir}")
        raise FileNotFoundError(
            f"Expected zero-padded frames like frame_0000.png in {frames_dir}"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    encoders = [
        ("libx264", ["-pix_fmt", "yuv420p"]),
        ("h264_nvenc", ["-pix_fmt", "yuv420p"]),
        ("h264_vaapi", []),
        ("mpeg4", []),
    ]

    last_error = ""
    for encoder, extra_args in encoders:
        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            encoder,
            *extra_args,
            output_path,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Video saved to {output_path} (encoder: {encoder})")
            return
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg is required to build videos from rollout frame images."
            ) from exc
        except subprocess.CalledProcessError as exc:
            last_error = exc.stderr or str(exc)

    raise RuntimeError(f"ffmpeg failed with all encoders:\n{last_error}")


def _build_video_from_npz(run_dir: str) -> None:
    """Legacy path: animate rollout_*.npz files saved by CNN rollout experiments."""
    rollout_files = glob.glob(os.path.join(run_dir, "rollout_*.npz"))
    if not rollout_files:
        raise FileNotFoundError(f"No rollout_*.npz files found in {run_dir}")

    videos_dir = os.path.join(run_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    for data_path in rollout_files:
        ic_name = os.path.basename(data_path).replace("rollout_", "").replace(".npz", "")
        print(f"Processing comparison video for initial condition: {ic_name}")

        data = np.load(data_path)
        rollout = data["rollout"]
        numerical = data["numerical"]

        fig, ax = plt.subplots(figsize=(10, 5))
        x_axis = np.arange(rollout.shape[1])
        ax.set_xlim(0, rollout.shape[1])
        y_min = float(np.min(numerical)) - 0.1
        y_max = float(np.max(numerical)) + 0.1
        ax.set_ylim(y_min, y_max)
        ax.grid(True)
        ax.set_xlabel("Spatial Grid")
        ax.set_ylabel("Amplitude")

        line_ana, = ax.plot([], [], lw=6, color="green", alpha=0.35, linestyle="-", label="Analytical")
        line_num, = ax.plot([], [], lw=2, color="blue", label="Numerical")
        line_ml, = ax.plot([], [], lw=2, color="red", linestyle="--", label="ML Prediction")
        ax.legend(loc="upper right")

        def init():
            line_ana.set_data([], [])
            line_num.set_data([], [])
            line_ml.set_data([], [])
            return line_ana, line_num, line_ml

        def animate(i):
            y_ana = np.roll(numerical[0], shift=2 * i)
            y_num = numerical[i]
            y_ml = rollout[i]
            line_ana.set_data(x_axis, y_ana)
            line_num.set_data(x_axis, y_num)
            line_ml.set_data(x_axis, y_ml)
            l2_err = np.sqrt(np.mean((y_num - y_ml) ** 2))
            ax.set_title(f"Comparison ({ic_name}) | Timestep: {i} | L2 Error: {l2_err:.5f}")
            return line_ana, line_num, line_ml

        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=rollout.shape[0],
            interval=20,
            blit=True,
        )
        save_path = os.path.join(videos_dir, f"comparison_{ic_name}.mp4")
        anim.save(save_path, writer="ffmpeg", fps=30)
        print(f"Video saved to {save_path}")
        plt.close(fig)


def _resolve_legacy_run_dir(user_path: str | None) -> str:
    if user_path:
        return user_path

    all_runs: list[str] = []
    for category in ["single", "sweeps"]:
        cat_dir = os.path.join(RUNS_BASE_DIR, category)
        if os.path.exists(cat_dir):
            for name in os.listdir(cat_dir):
                full_path = os.path.join(cat_dir, name)
                if os.path.isdir(full_path):
                    all_runs.append(full_path)

    if not all_runs:
        raise FileNotFoundError(f"No run directories found in {RUNS_BASE_DIR}/")

    all_runs.sort(key=os.path.getmtime)
    run_dir = all_runs[-1]
    print(f"Automatically selected latest run: {os.path.basename(run_dir)}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rollout comparison videos.")
    parser.add_argument(
        "path",
        nargs="?",
        help="Rollout frames directory or legacy ML run directory",
    )
    parser.add_argument(
        "--output",
        help="Output MP4 path when animating a rollout_frames directory",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    if args.path and os.path.isdir(args.path):
        frame_files = glob.glob(os.path.join(args.path, "frame_*.png"))
        if frame_files:
            output_path = args.output or os.path.join(args.path, "rollout.mp4")
            print(f"Building video from {len(frame_files)} frames in {args.path}")
            _build_video_from_frames(args.path, output_path, fps=args.fps)
            return

        rollout_files = glob.glob(os.path.join(args.path, "rollout_*.npz"))
        if rollout_files:
            _build_video_from_npz(args.path)
            return

        raise FileNotFoundError(
            f"No frame_*.png or rollout_*.npz files found in {args.path}"
        )

    run_dir = _resolve_legacy_run_dir(args.path)
    _build_video_from_npz(run_dir)


if __name__ == "__main__":
    main()
