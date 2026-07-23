"""Per-pixel gamma correction (Fan et al. / Gapuz 2025), ported from the
original C++ snippet, applied in batch to every image in data/data.

Output mirrors the input folder structure under data_corrected/, preserving
filenames (and therefore sequence).
"""

import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "data_corrected"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def gamma_correct(image: np.ndarray) -> np.ndarray:
    """Apply per-pixel gamma correction to a single-channel uint8 image."""
    # R1: invert -> Gaussian blur -> mask
    inverted = 255 - image
    mask_gamma = cv2.GaussianBlur(inverted, (21, 21), 0)

    # R2: per-pixel gamma map -- lambda(i,j) = 2^((anchor - mask(i,j)) / 128)
    mask_f = mask_gamma.astype(np.float64)
    exponent = (80.0 - mask_f) / 128.0
    lam = np.exp(exponent * np.log(2.0))
    lam = np.maximum(lam, 0.1)

    # R3: apply correction directly to grayscale pixel values
    # O(i,j) = 255 * (I(i,j)/255)^lambda(i,j)
    image_f = image.astype(np.float64) / 255.0
    log_i = np.log(image_f, out=np.full_like(image_f, -np.inf), where=image_f > 0)
    corrected = np.exp(lam * log_i)
    corrected *= 255.0
    corrected = np.clip(corrected, 0.0, 255.0)
    corrected += 0.5
    return corrected.astype(np.uint8)


def main():
    if not INPUT_DIR.is_dir():
        raise SystemExit(f"Input folder not found: {INPUT_DIR}")

    image_paths = sorted(
        p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        raise SystemExit(f"No images found under: {INPUT_DIR}")

    for i, src_path in enumerate(image_paths, 1):
        rel_path = src_path.relative_to(INPUT_DIR)
        dst_path = OUTPUT_DIR / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[skip] failed to read: {src_path}")
            continue

        corrected = gamma_correct(image)
        cv2.imwrite(str(dst_path), corrected)
        print(f"[{i}/{len(image_paths)}] {rel_path}")

    print(f"Done. Wrote {len(image_paths)} images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
