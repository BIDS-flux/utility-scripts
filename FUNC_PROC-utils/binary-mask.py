#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:15:47 2024

@author:
"""
import nibabel as nib
import numpy as np
from bids import BIDSLayout
import glob


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import nibabel as nib
from bids import BIDSLayout
import glob
import subprocess


# Define base directory
# CALGARY
bids_path = '/data/debug-proc/CPIP_bids'
tedana_dir = '/data/debug-proc/CPIP_bids/derivatives/tedana/'
subjects = "2532,3398,3136,2976,3328,2928"
ses = "1a"
bids_filters = {"task": ["laluna","partlycloudy", "rest"], "run": ["1", "2", None]}
# TORONTO
bids_path = '/data/debug-proc/CPIP_fMRI_July042025/CPIP_fMRI_July042025/'
tedana_dir = '/data/debug-proc/CPIP_fMRI_July042025/CPIP_fMRI_July042025/derivatives/tedana/'
subjects = "3536,3675,3504,3592,3803"
ses = None
bids_filters = {"task": ["laluna","partlycloudy", "rest"], "run": ["1", "2", "3", None]}

# Read the .bidsignore file (if it exists)
bidsignore_path = os.path.join(bids_path, ".bidsignore")
ignore_patterns = []

if os.path.exists(bidsignore_path):
    with open(bidsignore_path, "r") as f:
        ignore_patterns = [
            line.strip() for line in f if line.strip() and not line.strip().startswith("#")
        ]

expanded_ignores = []

for pattern in ignore_patterns:
    # If the pattern contains shell wildcards, resolve them
    if any(char in pattern for char in "*?[]"):
        full_pattern = os.path.join(bids_path, pattern)
        # Use recursive=True so "**" patterns are handled as well
        matches = glob.glob(full_pattern, recursive=True)

        # Keep paths relative to the BIDS root, just like the originals
        expanded_ignores.extend(
            os.path.relpath(p, bids_path) for p in matches
        )
    else:
        expanded_ignores.append(pattern)

# Remove duplicates while preserving order (Python ≥3.7 keeps dict order)
ignore_patterns = list(dict.fromkeys(expanded_ignores))

# Get the bids layout of the tedana dir dataset
tedana_layout = BIDSLayout(tedana_dir, validate=False, derivatives=True, ignore=ignore_patterns)

sessions = ses.split(",") if ses else [None]

for sub in subjects.split(','):
    for ses in sessions:
        nii_files = []

        for nii_f in tedana_layout.get(**bids_filters, subject=sub, session=ses, datatype='func', suffix='bold', extension='.nii.gz', space='MNI152NLin2009cAsym'):
            if "desc-denoisedREGRESSEDFltNewMcensored_" in nii_f.filename:
                nii_files.append([[nii_f.path],nii_f.path.replace('bold.nii.gz','mask.nii.gz')])
                print(f"#### Getting {[nii_f.path]} and {nii_f.path.replace('bold.nii.gz','mask.nii.gz')}")

        # List of input-output file pairs
        # Reference time point
        ref_timepoint = 80

        # Loop over each input-output file pair
        for nifti_files, output_file in nii_files:
            combined_mask, affine = create_combined_mask(nifti_files, ref_timepoint)
            save_nifti(combined_mask, affine, output_file)
            print(f"Combined mask saved to {output_file}")

def load_nifti(file_path):
    """Load a NIfTI file and return the image data and the affine."""
    img = nib.load(file_path)
    return img.get_fdata(), img.affine

def save_nifti(data, affine, file_path):
    """Save the data to a NIfTI file with the given affine."""
    img = nib.Nifti1Image(data, affine)
    nib.save(img, file_path)

def create_combined_mask(nifti_files, ref_timepoint=100):
    """Create a combined mask from the given NIfTI files using a specific time point."""
    combined_mask = None
    target_shape = None

    for i, file in enumerate(nifti_files):
        data, affine = load_nifti(file)
        
        if data.shape[-1] <= ref_timepoint:
            raise ValueError(f"File {file} does not have enough time points. Required: {ref_timepoint + 1}, Available: {data.shape[-1]}")
        
        ref_data = data[..., ref_timepoint]  # Extract the reference volume
        
        if target_shape is None:
            target_shape = ref_data.shape  # Set the target shape to the shape of the first reference volume
        
        # Ensure the current file's shape matches the target shape
        assert ref_data.shape == target_shape, f"File {file} does not match the target shape {target_shape}."
        
        # Create a binary mask for the reference volume
        mask = ref_data > 500  # Create a binary mask where values greater than 1 are 1

        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = np.logical_and(combined_mask, mask)  # Combine masks using logical AND

    return combined_mask.astype(np.uint8), affine