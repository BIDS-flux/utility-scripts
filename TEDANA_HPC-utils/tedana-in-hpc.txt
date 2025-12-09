#!/usr/bin/env bash
set -euo pipefail

# Input arguments
export BIDS_SUB_SES_DIR="${1}"   # e.g., sub-001_ses-01 or sub-001
export FMRIPREP_DIR="${2}"

# Output prefix directory
export out_prefix="${BIDS_SUB_SES_DIR}/func/"

# Loop over echo-1 files
for f in "${FMRIPREP_DIR}/${BIDS_SUB_SES_DIR}/func/"*_echo-1_*desc-preproc_bold.nii.gz; do
    [ -e "$f" ] || continue

    # Build prefixes
    in_prefix="${f%_echo-1_*desc-preproc_bold.nii.gz}"
    file_prefix="${in_prefix##*/func/}"

    mkdir -p "${out_prefix}"

    # Extract echo times (ms)
    echo_times=$(jq -j '.EchoTime*1000|tostring + " "' $(ls "${in_prefix}"_echo-*_*.json | sort))

    # Compute tedpca value (half of TR-based acquisition duration)
    tedpca_value=$(jq -r '(.AcquisitionDuration / .RepetitionTime / 2) | floor' "${in_prefix}"_echo-1_*preproc_bold.json)

    # Locate the corresponding brain mask
    mask_nii=$(compgen -G "${in_prefix}"'*'_desc-brain_mask.nii.gz | grep -v space || true)

    # Build TEDANA_OPTIONS dynamically (note: single $ so eval expands variables)
    export TEDANA_OPTIONS="--fittype curvefit \
        --ica-method robustica \
        --n-robust-runs 50 \
        --tree code/tedana_decision_tree.json \
        --external ${in_prefix}_desc-confounds_timeseries.tsv"

    echo "## Running tedana on ${in_prefix}"

    # Run tedana via eval (preserved)
    eval "tedana \
        --out-dir \"${out_prefix}\" \
        --prefix \"${file_prefix}\" \
        --mask \"${mask_nii}\" \
        -d ${in_prefix}_echo-*_desc-preproc_bold.nii.gz \
        -e ${echo_times} \
        --tedpca ${tedpca_value} \
        ${TEDANA_OPTIONS}"
done