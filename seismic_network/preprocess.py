#!/usr/bin/env python3
"""Convert SAC seismic files into readable CSV files.

The script scans the current folder for .sac files, reads the SAC binary
header and waveform samples, and writes each trace to a CSV file in a
separate folder named 'converted_sac'.
"""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path
from typing import List, Tuple

HEADER_SIZE = 632
DATA_OFFSET = 632


def detect_endian(header_bytes: bytes) -> Tuple[str, float, int]:
    """Infer the byte order and read the basic header values."""
    for endian in (">", "<"):
        try:
            delta = struct.unpack(endian + "f", header_bytes[0:4])[0]
            npts = struct.unpack(endian + "i", header_bytes[320:324])[0]
            if npts > 0 and abs(delta) < 1_000_000:
                return endian, delta, npts
        except struct.error:
            continue
    raise ValueError("Unable to parse SAC header; unsupported format.")


def read_sac(path: Path) -> dict:
    """Read a SAC file and return waveform metadata plus samples."""
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        raise ValueError(f"{path.name} is too small to be a SAC file")

    header_bytes = data[:HEADER_SIZE]
    endian, delta, npts = detect_endian(header_bytes)

    expected_data_bytes = npts * 4
    waveform_bytes = data[HEADER_SIZE : HEADER_SIZE + expected_data_bytes]
    if len(waveform_bytes) < expected_data_bytes:
        raise ValueError(f"{path.name} does not contain enough waveform data")

    samples = list(struct.unpack(endian + str(npts) + "f", waveform_bytes))

    return {
        "filename": path.name,
        "delta": delta,
        "npts": npts,
        "samples": samples,
    }


def write_csv(output_path: Path, trace: dict) -> None:
    """Write waveform samples to a CSV file with time and amplitude columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "amplitude"])
        for index, value in enumerate(trace["samples"]):
            time_value = index * trace["delta"]
            writer.writerow([time_value, value])


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "converted_sac"
    output_dir.mkdir(exist_ok=True)

    sac_files: List[Path] = []
    for pattern in ("*.sac", "*.SAC"):
        sac_files.extend(base_dir.glob(pattern))

    if not sac_files:
        print("No .sac files found in the current folder.")
        return

    manifest = []
    for sac_path in sorted(sac_files):
        try:
            trace = read_sac(sac_path)
            output_csv = output_dir / f"{sac_path.stem}.csv"
            write_csv(output_csv, trace)
            manifest.append(
                {
                    "source": sac_path.name,
                    "output": output_csv.name,
                    "npts": trace["npts"],
                    "delta": trace["delta"],
                }
            )
            print(f"Converted {sac_path.name} -> {output_csv.name}")
        except Exception as exc:
            print(f"Failed to convert {sac_path.name}: {exc}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Outputs saved in {output_dir}")


if __name__ == "__main__":
    main()
