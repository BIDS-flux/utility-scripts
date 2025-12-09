#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from pathlib import Path
import pydicom
from collections import defaultdict


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def as_int(value):
    """Robustly parse various DICOM value types to int; return None on failure."""
    if value is None:
        return None
    try:
        if isinstance(value, (bytes, bytearray)):
            value = value.decode(errors="ignore")
        return int(str(value).strip())
    except Exception:
        return None


def collect_series_files(input_dir: Path):
    """Return {series_number: [file paths]} mapping for all DICOMs in the directory."""
    series_map = defaultdict(list)

    for root, _, files in os.walk(input_dir):
        for fname in files:
            fpath = Path(root) / fname
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            except Exception:
                continue
            series_number = getattr(ds, "SeriesNumber", None)
            if series_number is not None:
                series_map[series_number].append(fpath)

    return series_map


def check_series_slices(series_number: int, dicom_files: list, log: logging.Logger) -> None:
    """Check expected vs found slices for a given series."""
    if not dicom_files:
        log.error("No files found for SeriesNumber %s", series_number)
        return

    # Read one header (no pixels) to pull the tags we need
    dcm = pydicom.dcmread(dicom_files[0], stop_before_pixels=True, force=True)

    # (0021,104F): expected slices per volume (often vendor/private; e.g., GE)
    tag_104f = dcm.get((0x0021, 0x104F))
    expected_slices_per_volume = as_int(tag_104f.value) if tag_104f is not None else None

    # (0020,1002): Images in Acquisition (total number of slices/images for the series)
    tag_1002 = dcm.get((0x0020, 0x1002))
    total_number_of_slices_in_series = as_int(tag_1002.value) if tag_1002 is not None else None

    found_slices = len(dicom_files)

    log.info("Series %s", series_number)
    log.info("  Found files in series:                %d", found_slices)

    if total_number_of_slices_in_series is not None:
        log.info("  Total slices expected (0020,1002):    %d", total_number_of_slices_in_series)
        if found_slices == total_number_of_slices_in_series:
            log.info("  ✅ Found count matches (0020,1002).")
        else:
            log.error(
                "  ❌ Mismatch vs (0020,1002): found %d, header says %d.",
                found_slices, total_number_of_slices_in_series
            )
    else:
        log.warning("  (0020,1002) Images in Acquisition not present or not parseable.")

    if expected_slices_per_volume is not None:
        log.info("  Slices per volume (0021,104F):        %d", expected_slices_per_volume)
        remainder = found_slices % expected_slices_per_volume
        if remainder == 0:
            n_vols = found_slices // expected_slices_per_volume
            log.info("  ✅ Complete volumes: %d full volume(s).", n_vols)
        else:
            log.error(
                "  ❌ Incomplete volumes: %d files not divisible by %d (remainder=%d).",
                found_slices, expected_slices_per_volume, remainder
            )
    else:
        log.warning("  (0021,104F) expected slices/volume not present or not parseable.")


def main():
    parser = argparse.ArgumentParser(
        description="Check DICOM series for slice completeness."
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
        help="Specific SeriesNumber to check. If omitted, all series are checked.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO).",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"Error: {args.input_dir} does not exist.")

    setup_logging(getattr(logging, args.log_level.upper()))
    log = logging.getLogger("dicom-check")

    series_map = collect_series_files(args.input_dir)
    if not series_map:
        log.error("No DICOM files found in %s", args.input_dir)
        sys.exit(1)

    if args.series is not None:
        dicom_files = series_map.get(args.series, [])
        check_series_slices(args.series, dicom_files, log)
    else:
        for series_number, files in sorted(series_map.items()):
            check_series_slices(series_number, files, log)


if __name__ == "__main__":
    main()
