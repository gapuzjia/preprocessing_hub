"""Compile a folder of images into a video, in filename order.

Usage:
    python make_video.py <input_dir> [output.mp4] [--fps 20]

Examples:
    python make_video.py data_corrected
    python make_video.py data_corrected\\data corrected.mp4
    python make_video.py data raw.mp4 --fps 20
"""

import argparse
import sys
from pathlib import Path

import cv2

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Folder of images (searched recursively)")
    parser.add_argument("output", type=Path, nargs="?", default=None, help="Output video path (default: <input_dir>.mp4)")
    parser.add_argument("--fps", type=float, default=20, help="Frames per second (default: 20)")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input folder not found: {args.input_dir}")

    image_paths = sorted(
        p for p in args.input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        sys.exit(f"No images found under: {args.input_dir}")

    output_path = args.output or args.input_dir.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        sys.exit(f"Failed to read: {image_paths[0]}")
    height, width = first.shape[:2]
    is_color = first.ndim == 3

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height), isColor=is_color)
    if not writer.isOpened():
        sys.exit(f"Failed to open video writer for: {output_path}")

    for i, path in enumerate(image_paths, 1):
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            print(f"[skip] failed to read: {path}")
            continue
        if frame.ndim != (3 if is_color else 2):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR if is_color else cv2.COLOR_BGR2GRAY)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        print(f"[{i}/{len(image_paths)}] {path.name}")

    writer.release()
    print(f"Done. Wrote {len(image_paths)} frames at {args.fps} fps to {output_path}")


if __name__ == "__main__":
    main()
