#!/usr/bin/env python3
"""
Check a single DICOM series for missing slices.

Usage
-----
python check_dicom_series.py <input_dir> --series 12
"""

import argparse
import os
import sys
from pathlib import Path

import pydicom


def check_dicom_series(input_dir: Path, series_number: int) -> None:
    """
    Walk `input_dir`, collect InstanceNumbers for the requested series,
    and report any gaps.

    Parameters
    ----------
    input_dir : Path
        Top‑level directory containing DICOM files.
    series_number : int
        The SeriesNumber you want to inspect.
    """
    instance_numbers = []

    for root, _, files in os.walk(input_dir):
        for fname in files:
            fpath = Path(root) / fname

            # Skip obviously non‑DICOM files early
            if not fpath.suffix and fpath.is_file():
                # anonymous DICOM often has no extension; allow it
                pass
            elif fpath.suffix.lower() not in {".dcm", ".dicom", ""}:
                continue

            try:
                ds = pydicom.dcmread(
                    fpath,
                    force=True,
                    stop_before_pixels=True,  # speed: we don't need pixel data
                )
            except Exception as e:
                print(f"⚠️  Skipping unreadable file {fpath}: {e}")
                continue

            # Filter by SeriesNumber
            if getattr(ds, "SeriesNumber", None) != series_number:
                continue

            # Collect InstanceNumber if present
            try:
                instance_numbers.append(int(ds.InstanceNumber))
            except AttributeError:
                print(f"⚠️  {fpath} has no InstanceNumber; skipping it.")

    if not instance_numbers:
        print(f"No DICOMs with SeriesNumber={series_number} were found.")
        return

    instance_numbers.sort()
    first, last = instance_numbers[0], instance_numbers[-1]
    missing = set(range(first, last + 1)) - set(instance_numbers)

    print(f"\nSeriesNumber {series_number}")
    print(f"Found {len(instance_numbers)} slices "
          f"(InstanceNumber {first}–{last}).")

    if missing:
        print(f"Missing slices: {sorted(missing)}")
    else:
        print("✅ No missing slices detected.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a DICOM series for missing slices."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the root folder containing DICOM files.",
    )
    parser.add_argument(
        "--series",
        "-s",
        type=int,
        required=True,
        dest="series_number",
        help="SeriesNumber to inspect.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"Error: {args.input_dir} does not exist.")

    check_dicom_series(args.input_dir, args.series_number)


if __name__ == "__main__":
    main()
