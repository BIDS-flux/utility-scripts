#!/usr/bin/env python3
"""
List unique In‑Stack Position Numbers for all DICOMs in a folder
that match a specified SeriesNumber.

Usage:
    python get_instack_positions.py /path/to/dicoms --series-number 3

Arguments:
    dicom_dir (positional): Path to a directory containing DICOM files (recursively scanned).
    --series-number (optional): Integer value of the SeriesNumber to filter on. Default is 3.
"""

import os
import glob
import argparse
import pydicom
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="List unique (0020,9057) In‑Stack Position Numbers from DICOMs for a given SeriesNumber"
    )
    parser.add_argument("dicom_dir", help="Folder containing DICOM files (will search recursively)")
    parser.add_argument(
        "--series-number", "-s",
        type=int,
        required=True,
        help="SeriesNumber to match (default: 3)"
    )
    args = parser.parse_args()

    instack_values = set()
    count_total = 0
    count_matched = 0

    for filepath in glob.iglob(os.path.join(args.dicom_dir, "**"), recursive=True):
        if os.path.isdir(filepath):
            continue  # skip directories

        count_total += 1
        try:
            ds = pydicom.dcmread(
                filepath,
                stop_before_pixels=True,
                specific_tags=["SeriesNumber", (0x0020, 0x9057)],
            )
        except Exception as e:
            log.debug(f"Skipping unreadable file '{filepath}': {e}")
            continue

        if getattr(ds, "SeriesNumber", None) == args.series_number:
            elem = ds.get((0x0020, 0x9057))
            if elem is not None:
                instack_values.add(int(elem.value))
                count_matched += 1

    log.info(f"Scanned {count_total} files.")
    log.info(f"Found {count_matched} matching SeriesNumber: {args.series_number}")
    log.info(f"Unique In-Stack Position Numbers:")

    for val in sorted(instack_values):
        log.info(f"  {val}")


if __name__ == "__main__":
    main()
