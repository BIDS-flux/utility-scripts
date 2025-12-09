#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 05 2025

@author: Milton Camacho
"""

import numpy as np
import os
import nibabel as nib
from nilearn import image as nimg
from nilearn import plotting as nplot
from bids import BIDSLayout
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Define base directory
# CALGARY
bids_path = '/data/debug-proc/CPIP_bids'
tedana_dir = '/data/debug-proc/CPIP_bids/derivatives/tedana/'
fmriprep_dir = '/data/debug-proc/CPIP_bids/derivatives/fmriprep2/'
subjects = "2532,3398,3136,2976,3328,2928"
ses = "1a"
bids_filters = {"task": ["laluna","partlycloudy", "rest"], "run": ["1", "2", None]}
# TORONTO
bids_path = '/data/debug-proc/CPIP_fMRI_July042025/CPIP_fMRI_July042025/'
tedana_dir = '/data/debug-proc/CPIP_fMRI_July042025/CPIP_fMRI_July042025/derivatives/tedana/'
fmriprep_dir = '/data/debug-proc/CPIP_fMRI_July042025/CPIP_fMRI_July042025/derivatives/fmriprep2/'
subjects = "3536,3675,3504,3592,3803"
ses = None
bids_filters = {"task": ["laluna","partlycloudy", "rest"], "run": ["1", "2", "3", None]}

# ATLAS
parcellation_mask_path = '/home/milton.camachocamach/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'

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

#read the parcellation atlas
parcellation_mask = nib.load(parcellation_mask_path)

for sub in subjects.split(','):
    for ses in sessions:
        nii_files = []
        mask_files = []

        for nii_f in tedana_layout.get(**bids_filters, subject=sub, session=ses, datatype='func', suffix='bold', extension='.nii.gz', space='MNI152NLin2009cAsym'):
            if "desc-denoisedREGRESSEDFltNewMcensored_" in nii_f.filename:
                nii_files.append(nii_f)
        
        for nii_f in tedana_layout.get(**bids_filters, subject=sub, session=ses, datatype='func', suffix='mask', extension='.nii.gz', space='MNI152NLin2009cAsym'):
            if "desc-denoisedREGRESSEDFltNewMcensored_" in nii_f.filename:
                mask_files.append(nii_f)

        # Sort nii_files and mask_files by their filename attribute
        nii_files.sort(key=lambda x: x.filename)
        mask_files.sort(key=lambda x: x.filename)
        
        sorted_nii_files = [[nii, mask] for nii, mask in zip(nii_files, mask_files)]

        bids_entities = {}

        for nii_file,mask_file in sorted_nii_files:
            # Extract BIDS entities from the NIfTI file
            bids_entities['task'] = nii_file.entities.get('task', None)
            bids_entities['run'] = nii_file.entities.get('run', None)
            bids_entities['acquisition'] = nii_file.entities.get('acquisition', None)

            # Load the functional image and brain mask
            func_img = nib.load(nii_file.path)
            brain_mask_img = nib.load(mask_file.path)

            # Resample parcellation to functional image
            resampled_parcellation = nimg.resample_to_img(parcellation_mask, func_img, interpolation='nearest')

            # Extract a single timepoint (e.g., the first timepoint, index 0)
            func_3d = func_img.slicer[..., 0]

            # Plot the 3D functional image with the parcellation overlay
            nplot.plot_img(func_3d, title="Functional Image with Parcellation Overlay", bg_img=None, draw_cross=True, cut_coords=(0, -30, 30))

            plot_title = f'Parcellation Overlay for {nii_file.filename}'
            nplot.plot_roi(resampled_parcellation, bg_img=func_3d, title=plot_title, draw_cross=True)

            # Ensure the output directory exists for saving
            overlay_image_path = nii_file.path.replace('_bold.nii.gz','_parcellation-overlay.png')
            os.makedirs(os.path.dirname(overlay_image_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(overlay_image_path)
            plt.show()
            plt.close()

            # Save the resampled parcellation atlas
            resampled_parcellation.to_filename(nii_file.path.replace('desc-denoisedREGRESSEDFltNewMcensored_bold','desc-censoredSchaefer2018_parcellation'))

            # Get data arrays
            func_data = func_img.get_fdata()
            parcellation_data = resampled_parcellation.get_fdata()
            brain_mask_data = brain_mask_img.get_fdata() > 0

            # Combine masks
            combined_mask = (parcellation_data > 0) & brain_mask_data

            # Get unique ROI labels
            roi_labels = np.unique(parcellation_data[combined_mask])

            # Initialize DataFrame for timeseries
            timeseries_data = pd.DataFrame()

            # Extract timeseries for each ROI
            for roi_label in roi_labels:
                roi_mask = (parcellation_data == roi_label) & combined_mask
                roi_timeseries = func_data[roi_mask, :].mean(axis=0)
                timeseries_data[f'ROI_{int(roi_label)}'] = roi_timeseries

            # Save timeseries to CSV
            output_csv_path = nii_file.path.replace('desc-denoisedREGRESSEDFltNewMcensored_bold.nii.gz','desc-censoredSchaefer2018_timeseries.tsv')
            timeseries_data.to_csv(output_csv_path, index=False, sep='\t')

            # Compute connectome matrix
            connectome_matrix = timeseries_data.corr()
            connectome_matrix_path = nii_file.path.replace('desc-denoisedREGRESSEDFltNewMcensored_bold.nii.gz','desc-censoredSchaefer2018_connectome.tsv')
            connectome_matrix.to_csv(connectome_matrix_path, index=True, sep='\t')

            # Plot connectome matrix
            plt.figure(figsize=(12, 10))
            plt.imshow(connectome_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.colorbar(label='Correlation')
            plt.title(f'Connectome Matrix for subject: {sub} session: {ses} task: {bids_entities["task"]} run: {bids_entities["run"]}')
            plt.xlabel('ROI')
            plt.ylabel('ROI')
            plt.tight_layout()

            connectome_plot_path = nii_file.path.replace('desc-denoisedREGRESSEDFltNewMcensored_bold.nii.gz','desc-censoredSchaefer2018_connectome.png')
            plt.savefig(connectome_plot_path)
            plt.show()
            plt.close()

            print(f"Connectome plot saved to: {connectome_plot_path}")

            # Plot timeseries
            plt.figure(figsize=(20, 10))
            for column in timeseries_data.columns:
                plt.plot(timeseries_data[column], label=column)

            plt.title(f'Timeseries for subject: {sub} session: {ses} task: {bids_entities["task"]} run: {bids_entities["run"]}')
            plt.xlabel('Timepoints')
            plt.ylabel('Signal Intensity')
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            plt.tight_layout()

            timeseries_plot_path = nii_file.path.replace('desc-denoisedREGRESSEDFltNewMcensored_bold.nii.gz','desc-censoredSchaefer2018_timeseries.png')

            plt.savefig(timeseries_plot_path)
            plt.show()
            plt.close()

            print(f"Timeseries plot saved to: {timeseries_plot_path}")
            print(f"Connectome matrix saved to: {connectome_matrix_path}")