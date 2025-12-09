#!/usr/bin/env python3
"""
Copy complete 4‑D DICOM volumes and skip incomplete ones.

This script reads DICOM files from an input directory, applies optional header filters,
groups slices by SeriesInstanceUID and inferred volume index, and copies only complete
volumes (i.e., those with the expected number of slices) to the output folder.

Optionally, it can delete skipped files from a given folder.

Usage:
    python script.py input_folder output_folder [--expected_slices_per_volume N]
                                                [--remove_skipped_from PATH]
                                                [--filter KEY=VALUE ...]
"""

import os
import glob
import pydicom
import shutil
from collections import defaultdict
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def _parse_filters(raw_list):
    """Parse repeated KEY=VALUE filters from CLI input."""
    filt = {}
    if raw_list:
        for kv in raw_list:
            if "=" not in kv:
                raise argparse.ArgumentTypeError(f"--filter '{kv}' is not KEY=VALUE")
            k, v = kv.split("=", 1)
            filt[k] = v
    return filt


def process_dicom_files(
    input_folder,
    output_folder,
    expected_slices_per_volume_user_specified=None,
    remove_skipped_from=None,
    filters=None,
):
    """Process and copy complete DICOM volumes from input to output folder."""
    filters = filters or {}
    total_number_of_slices_in_series = 0

    output_dir = os.path.join(output_folder, "complete_volumes")
    os.makedirs(output_dir, exist_ok=True)

    dicom_files = glob.glob(os.path.join(input_folder, "*"))
    log.info(f"Found {len(dicom_files)} DICOM files in '{input_folder}'")

    grouped_files = defaultdict(lambda: defaultdict(list))
    wanted = 0

    for f in dicom_files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception as e:
            log.warning(f"⚠️  Skipping unreadable '{f}': {e}")
            continue

        for k, want in filters.items():
            if str(getattr(dcm, k, "")) != want:
                break
        else:
            wanted += 1
            series_uid = dcm.SeriesInstanceUID
            z_position = getattr(dcm, "ImagePositionPatient", [None, None, None])[2]
            instance_number = getattr(dcm, "InstanceNumber", None)
            in_stack_elem = dcm.get((0x0020, 0x9057))  # In‑Stack Position Number
            expected_slices_per_volume = dcm.get((0x0021, 0x104F)).value
            in_stack_pos = in_stack_elem.value if in_stack_elem else None
            total_number_of_slices_in_series = dcm.get((0x0020, 0x1002)).value

            if expected_slices_per_volume_user_specified is not None:
                expected_slices_per_volume = expected_slices_per_volume_user_specified

            if None in (z_position, instance_number, in_stack_pos):
                log.warning(
                    f"⚠️  {f} has incomplete header; skipping. "
                    f"Missing: z_position={z_position}, instance_number={instance_number}, in_stack_pos={in_stack_pos}"
                )
                continue

            z_position = round(float(z_position), 3)
            volume_index = (int(instance_number - 1) // int(expected_slices_per_volume)) + 1

            grouped_files[series_uid][volume_index].append(
                (z_position, int(instance_number), int(in_stack_pos), f)
            )

    if wanted != total_number_of_slices_in_series:
        log.warning(
            f"⚠️  Found {wanted} matching files for filters={filters}, but expected {total_number_of_slices_in_series}"
        )
    else:
        log.info(
            f"✅ Found {wanted} files for filters={filters}; matches expected {total_number_of_slices_in_series}"
        )

    skipped_files = []

    for series_uid, volumes in grouped_files.items():
        for vol_idx, files in volumes.items():
            files.sort(key=lambda x: x[0])

            complete = len(files) == expected_slices_per_volume
            if not complete:
                log.warning(
                    f"Skipping incomplete volume (Series: {series_uid}, Vol: {vol_idx}) "
                    f"- found {len(files)}/{expected_slices_per_volume} slices"
                )
                skipped_files.extend(fp for *_, fp in files)
                continue

            for *_, file_path in files:
                shutil.copyfile(
                    file_path,
                    os.path.join(output_dir, os.path.basename(file_path)),
                )

    if remove_skipped_from and skipped_files:
        for fp in skipped_files:
            try:
                os.remove(os.path.join(remove_skipped_from, os.path.basename(fp)))
                log.info(f"Removed skipped file: {fp}")
            except FileNotFoundError:
                log.warning(f"File not found for removal: {fp}")

    log.info(f"Copied all complete volumes into '{output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy complete 4‑D DICOM volumes; skip incomplete ones."
    )
    parser.add_argument("input_folder", help="Folder containing DICOM files")
    parser.add_argument("output_folder", help="Destination folder")
    parser.add_argument(
        "--expected_slices_per_volume",
        "-s",
        default=None,
        help="Override number of slices per volume (from tag (0021,104F))",
    )
    parser.add_argument(
        "--remove_skipped_from",
        help="Also delete skipped slices from this folder",
        default=None,
    )
    parser.add_argument(
        "--filter",
        action="append",
        metavar="KEY=VALUE",
        help="Header filter; repeatable (e.g. --filter SeriesNumber=3)",
    )

    args = parser.parse_args()
    process_dicom_files(
        args.input_folder,
        args.output_folder,
        args.expected_slices_per_volume,
        args.remove_skipped_from,
        filters=_parse_filters(args.filter),
    )
