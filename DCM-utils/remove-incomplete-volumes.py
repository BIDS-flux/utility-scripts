#!/usr/bin/env python3
"""
Copy complete 4‑D DICOM volumes and skip incomplete ones.

This script reads DICOM files from an input directory, applies optional DICOM tag filters, groups slices into volumes based on expected slice count, and copies only complete volumes into an output directory.

Usage:
    python script.py <input_folder> <output_folder> [--expected_slices_per_volume N] [--remove_skipped_from PATH] [--filter KEY=VALUE ...]

Arguments:
    input_folder: Directory containing DICOM files (flat or nested)
    output_folder: Directory to save complete volumes into
    --expected_slices_per_volume: Override the (0021,104F) tag with user-defined slice count per volume
    --remove_skipped_from: Remove incomplete/skipped DICOMs from this directory
    --filter: Header filter in the form KEY=VALUE (repeatable)
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


# ✨ 0) helper ────────────────────────────────────────────────────────────────
def _parse_filters(raw_list):
    filt = {}
    if raw_list:
        for kv in raw_list:
            if "=" not in kv:
                raise argparse.ArgumentTypeError(f"--filter '{kv}' is not KEY=VALUE")
            k, v = kv.split("=", 1)
            filt[k] = v
    return filt


# ✨ 1) added **filters** arg (default {})
def process_dicom_files(
    input_folder,
    output_folder,
    expected_slices_per_volume_user_specified=None,
    remove_skipped_from=None,
    filters=None,
):
    filters = filters or {}
    total_number_of_slices_in_series = 0

    # target directory for good slices
    output_dir = os.path.join(output_folder, "complete_volumes")
    os.makedirs(output_dir, exist_ok=True)

    dicom_files = glob.glob(os.path.join(input_folder, "*"))
    log.info(f"Found {len(dicom_files)} DICOM files in '{input_folder}'")

    # grouped_files[series_uid][volume_idx] →
    #       list[(z_pos, inst_no, in_stack_pos, path)]
    grouped_files = defaultdict(lambda: defaultdict(list))

    wanted = 0
    for f in dicom_files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception as e:
            log.warning(f"⚠️  Skipping unreadable file '{f}': {e}")
            continue

        # ✨ 2) header filters -------------------------------------------------
        for k, want in filters.items():
            if str(getattr(dcm, k, "")) != want:
                break
        else:
            # passed all filters
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
                    f"⚠️  {f} has incomplete header; skipping it. "
                    f"Missing: z_position={z_position}, instance_number={instance_number}, in_stack_pos={in_stack_pos}"
                )
                continue

            z_position = round(float(z_position), 3)

            # ✨ 3) volume index: 1‑based instance numbers ⇒ subtract 1
            volume_index = (int(instance_number - 1) // int(expected_slices_per_volume)) + 1

            grouped_files[series_uid][volume_index].append(
                (z_position, int(instance_number), int(in_stack_pos), f)
            )

    if wanted != total_number_of_slices_in_series:
        log.warning(
            f"Found {wanted} files matching filters={filters}, "
            f"but expected {total_number_of_slices_in_series}"
        )
    else:
        log.info(
            f"Found {wanted} files matching filters={filters}; expected "
            f"{total_number_of_slices_in_series} — all good!"
        )

    skipped_files = []

    # ────────── pass 2: validate & copy ─────────────────────────────────────
    for series_uid, volumes in grouped_files.items():
        for vol_idx, files in volumes.items():
            files.sort(key=lambda x: x[0])
            in_stack_positions = sorted({p for _, _, p, _ in files})

            complete = (
                len(in_stack_positions) == expected_slices_per_volume
                and in_stack_positions == list(range(1, expected_slices_per_volume + 1))
            )

            if not complete:
                log.warning(
                    f"Skipping incomplete volume "
                    f"(Series: {series_uid}, Vol: {vol_idx}) "
                    f"- found {len(in_stack_positions)}/{expected_slices_per_volume} "
                    f"unique In‑Stack positions: {in_stack_positions}"
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


# ─────────────────────────── CLI ────────────────────────────────────────────
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
        help="Override the number of slices per volume (from tag 0021,104F)",
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
