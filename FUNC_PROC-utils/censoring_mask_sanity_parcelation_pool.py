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
import ast
from decimal import Decimal
import subprocess
from nilearn.image import index_img
import json

#all "subjects": "2532,3398,3136,2976,3328,2928",

# Define site configurations as a dictionary
site_configs = {
    "calgary": {
        "bids_path": "/data/debug-proc/CPIP_bids",
        "tedana_dir": "/data/debug-proc/CPIP_bids/derivatives/tedan2/",
        "fmriprep_dir": "/data/debug-proc/CPIP_bids/derivatives/fmriprep2/",
        "subjects": "2532,3398,3136,2976,3328",
        "ses": "1a",
        "bids_filters": {"task": ["laluna", "partlycloudy", "rest"], "run": ["1", "2", None]}
    },
    "toronto": {
        "bids_path": "/data/debug-proc/CPIP_bids_toronto/",
        "tedana_dir": "/data/debug-proc/CPIP_bids_toronto/derivatives/tedana",
        "fmriprep_dir": "/data/debug-proc/CPIP_bids_toronto/derivatives/fmriprep",
        "subjects": "4645,4131,4097,3863,3901",
        "ses": "1a",
        "bids_filters": {"task": ["laluna", "partlycloudy", "rest"], "run": ["1", "2", "3", None]}
    },
    "montreal": {
        "bids_path": "/data/debug-proc/CPIP_bids_montreal/",
        "tedana_dir": "/data/debug-proc/CPIP_bids_montreal/derivatives/tedana2/",
        "fmriprep_dir": "/data/debug-proc/CPIP_bids_montreal/derivatives/fmriprep/",
        "subjects": "1118,1312,1760,2117,1755",
        "ses": "1a",
        "bids_filters": {"task": ["laluna", "partlycloudy", "rest"], "run": ["1", "2", "3", None]}
    }
}

# read the parcellation labels
parcellation_labels_path = "/data/debug-proc/Schaefer200_labels_7network.csv"

parcellation_labels = pd.read_csv(parcellation_labels_path,sep=',')

parcellation_labels = list(parcellation_labels["ROI Name"].values)

def _read_tsv_max_precision(path: str) -> pd.DataFrame:
    """
    Read TSV with ROI_* columns parsed through Decimal→float (faithful round-trip),
    leaving other columns as-is. Falls back to NaN on blanks.
    """
    # Grab header to detect ROI columns
    cols = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    roi_cols = [c for c in cols if c.startswith("ROI_")]

    # Read everything as string first to avoid premature float casting
    df = pd.read_csv(path, sep="\t", dtype=str)

    # Convert ROI columns through Decimal → float64 (IEEE-754 double)
    for c in roi_cols:
        df[c] = df[c].map(lambda x: float(Decimal(x)) if isinstance(x, str) and x != "" else np.nan)

    # (Optional) try to coerce any non-ROI numeric columns
    for c in df.columns:
        if c not in roi_cols:
            # best-effort numeric, but don't break non-numerics
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

# Read the .bidsignore file (if it exists)
def get_ignore_patterns(bids_path):
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
            matches = glob.glob(full_pattern, recursive=True)
            expanded_ignores.extend(
                os.path.relpath(p, bids_path) for p in matches
            )
        else:
            expanded_ignores.append(pattern)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(expanded_ignores))

def load_layouts(tedana_dir, fmriprep_dir, ignore):
    tedana_layout   = BIDSLayout(tedana_dir,   validate=False, derivatives=True, ignore=ignore)
    fmriprep_layout = BIDSLayout(fmriprep_dir, validate=False, derivatives=True, ignore=ignore)
    return tedana_layout, fmriprep_layout

def find_tedana_bold_files(layout, subject, session, filters):
    """Return list of BIDSFile objs for tedana outputs of interest."""
    hits = layout.get(
        **filters, subject=subject, session=session, datatype='func',
        suffix='bold', extension='.nii.gz', space='MNI152NLin2009cAsym'
    )
    # keep only the denoised + regressed images you target
    return [bf for bf in hits if "desc-denoisedREGRESSEDFlt_" in bf.filename]

def find_confounds_tsv(layout, subject, session, entities):
    """Return the single confounds tsv path matching task/run/acq."""
    files = layout.get(
        subject=subject, session=session, suffix='timeseries', extension='.tsv',
        desc='confounds', **{k: v for k, v in entities.items() if v is not None},
        return_type='filename'
    )
    if not files:
        return None
    if len(files) > 1:
        # Heuristic: prefer file with 'desc-confounds_timeseries.tsv' in name
        files = sorted(files, key=lambda p: ('desc-confounds_timeseries' not in p, p))
    return files[0]

def get_non_steady_indices(confounds):
    oh = confounds.filter(regex=r'^non_steady_state_outlier')
    if not oh.empty:
        return list(np.where(oh.any(axis=1))[0])
    return []

def get_motion_outlier_indices(confounds):
    oh = confounds.filter(regex=r'^motion_outlier')
    if not oh.empty:
        return list(np.where(oh.any(axis=1))[0])
    return []

def compute_and_save_tsnr_std(nifti_file: str):
    """
    Compute tSNR and std_t maps for a NIfTI file and return the NIfTI images and output file paths.
    """
    # Load the NIfTI file
    nifti_img = nib.load(nifti_file.path)
    nifti_data = nifti_img.get_fdata()
    eps = 1e-8 # avoid division by zero
    tsnr_data = np.mean(nifti_data, axis=-1) / (np.std(nifti_data, axis=-1) + eps)
    std_t_data = np.std(nifti_data, axis=-1)

    # Create NIfTI images
    tsnr_img = nib.Nifti1Image(tsnr_data, nifti_img.affine)
    std_t_img = nib.Nifti1Image(std_t_data, nifti_img.affine)

    # Generate output file paths
    tsnr_file = nifti_file.path.replace('_bold.', '_tsnr.')
    std_t_file = nifti_file.path.replace('_bold.', '_stdt.')

    # Save the NIfTI images
    nib.save(tsnr_img, tsnr_file)
    print(f"Saved tSNR image to {tsnr_file}")
    nib.save(std_t_img, std_t_file)
    print(f"Saved stdt image to {std_t_file}")

def apply_keep_to_image(img, keep_idx):
    return img if len(keep_idx) == img.shape[-1] else index_img(img, keep_idx)

def drop_indices(df, drop_idx):
    return df.drop(index=drop_idx).reset_index(drop=True)

def save_sidecar_indices(out_bold_path, motion_outliers_rel):
    sidecar = out_bold_path.replace('_bold.nii.gz', '_censoridx.json')
    payload = {
        "motion_outlier_idx": [int(i) for i in motion_outliers_rel]
    }
    with open(sidecar, 'w') as f:
        json.dump(payload, f, indent=2)
    return sidecar

def save_nifti(data, affine, file_path):
    """Save the data to a NIfTI file with the given affine."""
    img = nib.Nifti1Image(data, affine)
    nib.save(img, file_path)

def load_nifti(file_path):
    """Load a NIfTI file and return the image data and the affine."""
    img = nib.load(file_path)
    return img.get_fdata(), img.affine

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
        mask = ref_data > 1  # Create a binary mask where values greater than 1 are 1

        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = np.logical_and(combined_mask, mask)  # Combine masks using logical AND

    return combined_mask.astype(np.uint8), affine

#read the parcellation atlas
parcellation_mask_path = '/home/milton.camachocamach/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
parcellation_mask = nib.load(parcellation_mask_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main work
# ─────────────────────────────────────────────────────────────────────────────
# 1) Expand ignore patterns + build layouts
for site in site_configs:
    config = site_configs[site]
    bids_path = config["bids_path"]
    tedana_dir = config["tedana_dir"]
    fmriprep_dir = config["fmriprep_dir"]
    subjects = config["subjects"]
    ses = config["ses"]
    bids_filters = config["bids_filters"]

    ignore_patterns = get_ignore_patterns(bids_path)
    tedana_layout, fmriprep_layout = load_layouts(tedana_dir, fmriprep_dir, ignore_patterns)

    # 2) Sessions handling
    sessions = ses.split(",") if ses else [None]
    
    for sub in subjects.split(','):
        for ses in sessions:
            
            #################################################################
            # censoring
            #################################################################
            
            nii_files = []
                
            nii_files = find_tedana_bold_files(tedana_layout, sub, ses, bids_filters)

            for nifti_file in nii_files:
                entities = {
                    'task':        nifti_file.entities.get('task'),
                    'run':         nifti_file.entities.get('run'),
                    'acquisition': nifti_file.entities.get('acquisition'),
                }

                conf_tsv = find_confounds_tsv(fmriprep_layout, sub, ses, entities)
                if not conf_tsv:
                    print(f"[WARN] No confounds for {nifti_file.filename}. Skipping.")
                    continue

                if not (os.path.exists(nifti_file.path) and os.path.exists(conf_tsv)):
                    print("##############################################")
                    print(f"[WARN] Missing files for {nifti_file.filename}. Skipping.")
                    continue

                img = nib.load(nifti_file.path)
                conf = pd.read_csv(conf_tsv, sep='\t')

                current_desc = nifti_file.entities.get('desc', 'denoisedREGRESSEDFlt')
                out_bold = nifti_file.path.replace(current_desc, f"{current_desc}censored")

                # Step 1: Remove dummy volumes (non-steady-state)
                nonsteady_idx = get_non_steady_indices(conf)
                keep_idx = [i for i in range(len(conf)) if i not in nonsteady_idx]
                conf_clean = drop_indices(conf, nonsteady_idx)
                img_clean = apply_keep_to_image(img, keep_idx)

                # Step 2: Mark motion outliers relative to cleaned indices
                motion_idx_original = get_motion_outlier_indices(conf)
                motion_idx_rel = [keep_idx.index(i) for i in motion_idx_original if i in keep_idx]

                # Step 3: Drop dummy columns from confounds
                oh_cols = conf_clean.filter(regex=r'^non_steady_state_outlier').columns
                conf_clean = conf_clean.drop(columns=oh_cols, errors='ignore')

                # Step 4: Save outputs
                nib.save(img_clean, out_bold)
                out_conf = out_bold.replace('_bold.nii.gz', '_confounds.tsv')
                conf_clean.to_csv(out_conf, sep='\t', index=False)

                sidecar = save_sidecar_indices(out_bold, motion_idx_rel)

                print(f"[OK] Saved censored BOLD: {out_bold}")
                print(f"[OK] Saved pruned confounds: {out_conf}")
                print(f"[OK] Motion outliers marked in: {sidecar} ({len(motion_idx_rel)} TRs)")
            
            #################################################################
            # binary mask creation
            #################################################################

            tedana_layout = BIDSLayout(tedana_dir, validate=False, derivatives=True, ignore=ignore_patterns)

            nii_files = []

            for nii_f in tedana_layout.get(**bids_filters, subject=sub, session=ses, datatype='func', suffix='bold', extension='.nii.gz', space='MNI152NLin2009cAsym'):
                if "desc-denoisedREGRESSEDFltcensored_" in nii_f.filename:
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

            #################################################################
            # Sanity check
            #################################################################

            nii_files = []

            for nii_f in tedana_layout.get(**bids_filters, subject=sub, session=ses, datatype='func', suffix='bold', extension='.nii.gz', space=['MNI152NLin2009cAsym',None]):
                if "desc-denoisedREGRESSEDFlt_" in nii_f.filename:
                    nii_files.append(nii_f)
                if "desc-denoisedREGRESSEDFltcensored_" in nii_f.filename:
                    nii_files.append(nii_f)
                if "desc-optcom_bold" in nii_f.filename:
                    nii_files.append(nii_f)

            bids_entities = {}

            for nifti_file in nii_files:
                # Extract BIDS entities from the NIfTI file
                bids_entities['task'] = nifti_file.entities.get('task', None)
                bids_entities['run'] = nifti_file.entities.get('run', None)
                bids_entities['acquisition'] = nifti_file.entities.get('acquisition', None)

                # Check if files exist before proceeding
                if not os.path.exists(nifti_file.path):
                    print("##############################################")
                    print(f"Files for {series_folder} not found. Skipping this acquisition.")
                    continue
                
                compute_and_save_tsnr_std(nifti_file)

            #################################################################
            # parcellation
            #################################################################

            # Get the bids layout of the tedana dir dataset
            tedana_layout = BIDSLayout(tedana_dir, validate=False, derivatives=True, ignore=ignore_patterns)

            nii_files = []
            mask_files = []

            # get all bold.nii.gz files
            for nii_f in tedana_layout.get(**bids_filters, subject=sub, session=ses, datatype='func', suffix='bold', extension='.nii.gz', space='MNI152NLin2009cAsym'):
                if "desc-denoisedREGRESSEDFltcensored_" in nii_f.filename:
                    nii_files.append(nii_f)
            # get all mask.nii.gz files
            for nii_f in tedana_layout.get(**bids_filters, subject=sub, session=ses, datatype='func', suffix='mask', extension='.nii.gz', space='MNI152NLin2009cAsym'):
                if "desc-denoisedREGRESSEDFltcensored_" in nii_f.filename:
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
                resampled_parcellation.to_filename(nii_file.path.replace('desc-denoisedREGRESSEDFltcensored_bold','desc-censoredSchaefer2018_parcellation'))

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
                output_csv_path = nii_file.path.replace('desc-denoisedREGRESSEDFltcensored_bold.nii.gz','desc-censoredSchaefer2018_timeseries.tsv')
                timeseries_data['tr_number'] = np.arange(1, timeseries_data.shape[0]+1)
                timeseries_data.to_csv(output_csv_path, index=False, sep='\t')

                # Compute connectome matrix
                connectome_matrix = timeseries_data.corr()
                connectome_matrix_path = nii_file.path.replace('desc-denoisedREGRESSEDFltcensored_bold.nii.gz','desc-censoredSchaefer2018_connectome.tsv')
                connectome_matrix.to_csv(connectome_matrix_path, index=True, sep='\t')

                # Plot connectome matrix
                plt.figure(figsize=(12, 10))
                plt.imshow(connectome_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                plt.colorbar(label='Correlation')
                plt.title(f'Connectome Matrix for subject: {sub} session: {ses} task: {bids_entities["task"]} run: {bids_entities["run"]}')
                plt.xlabel('ROI')
                plt.ylabel('ROI')
                plt.tight_layout()

                connectome_plot_path = nii_file.path.replace('desc-denoisedREGRESSEDFltcensored_bold.nii.gz','desc-censoredSchaefer2018_connectome.png')
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

                timeseries_plot_path = nii_file.path.replace('desc-denoisedREGRESSEDFltcensored_bold.nii.gz','desc-censoredSchaefer2018_timeseries.png')

                plt.savefig(timeseries_plot_path)
                plt.show()
                plt.close()

                print(f"Timeseries plot saved to: {timeseries_plot_path}")
                print(f"Connectome matrix saved to: {connectome_matrix_path}")

#################################################################
# crate mega tsv
#################################################################

tsv_files = []

for site in site_configs:
    config = site_configs[site]
    bids_path   = config["bids_path"]
    tedana_dir  = config["tedana_dir"]
    # fmriprep_dir = config["fmriprep_dir"]   # not needed for this part
    subjects    = config["subjects"]
    ses         = config["ses"]
    bids_filters= config["bids_filters"]

    ignore_patterns = get_ignore_patterns(bids_path)
    tedana_layout = BIDSLayout(tedana_dir, validate=False, derivatives=True, ignore=ignore_patterns)

    sessions = ses.split(",") if ses else [None]
    for sub in subjects.split(','):
        for this_ses in sessions:
            print(f"Site: {site}, Subject: {sub}, Session: {this_ses}, Task: {bids_filters.get('task')}, Run: {bids_filters.get('run')}")
            # find censored Schaefer timeseries TSVs
            for tsv_f in tedana_layout.get(
                **bids_filters, subject=sub, session=this_ses,
                datatype='func', suffix='timeseries', extension='.tsv',
                space='MNI152NLin2009cAsym'
                ):
                if "desc-censoredSchaefer2018_" not in tsv_f.filename:
                    continue

                # get TR from matching desc-optcom bold JSON (if available)
                tr_value = None
                for nii_f in tedana_layout.get(
                    **bids_filters, subject=sub, session=this_ses,
                    datatype='func', suffix='bold', extension='.json'
                    ):
                    if "desc-optcom_bold.json" in nii_f.filename:  # fix: check nii_f, not tsv_f
                        tr_value = json.load(open(nii_f.path))['RepetitionTime']
                        break

                # append as a tuple (site, tsv_file, tr_value)
                tsv_files.append((site, tsv_f, tr_value))

for site in site_configs:
    config = site_configs[site]
    bids_path   = config["bids_path"]
    # fmriprep_dir = config["fmriprep_dir"]   # not needed for this part
    tedana_dir  = config["tedana_dir"]
    subjects    = config["subjects"]
    ses         = config["ses"]
    bids_filters= config["bids_filters"]

    ignore_patterns = get_ignore_patterns(bids_path)
    tedana_layout = BIDSLayout(tedana_dir, validate=False, derivatives=True, ignore=ignore_patterns)

    sessions = ses.split(",") if ses else [None]
    for sub in subjects.split(','):
        for this_ses in sessions:
            # find censored Schaefer timeseries TSVs
            print(f"Site: {site}, Subject: {sub}, Session: {this_ses}, Task: {bids_filters.get('task')}, Run: {bids_filters.get('run')}")
            for tsv_f in tedana_layout.get(
                **bids_filters, subject=sub, session=this_ses,
                datatype='func', suffix='timeseries', extension='.tsv',
                space='MNI152NLin2009cAsym'
            ):
                if "desc-censoredSchaefer2018_" not in tsv_f.filename:
                    print("valio verga")
                    continue

                # get TR from matching desc-optcom bold JSON (if available)
                tr_value = None
                for nii_f in tedana_layout.get(
                    **bids_filters, subject=sub, session=this_ses,
                    datatype='func', suffix='bold', extension='.json'
                ):
                    if "desc-optcom_bold.json" in nii_f.filename:  # fix: check nii_f, not tsv_f
                        tr_value = json.load(open(nii_f.path))['RepetitionTime']
                        break

                # append as a tuple (site, tsv_file, tr_value)
                print(str(site) + str(tsv_f) + str(tr_value))
                
                tsv_files.append((site, tsv_f, tr_value))

tsv_bids_files = {}

for site, tsv_file, tr_value in tsv_files:
    entities = tsv_file.entities
    subject  = entities.get('subject', None)
    session  = entities.get('session', None)
    task     = entities.get('task', None)
    run      = entities.get('run', None)

    key = (subject, session, task, run, site)

    # >>> read with max precision helper
    tsv_df = _read_tsv_max_precision(tsv_file.path)

    # >>> add per-file 0-based volume index (count of volumes in the 4D image)
    # (no NumPy needed)
    tsv_df.insert(0, 'vol_idx', range(len(tsv_df)))

    if key not in tsv_bids_files:
        tsv_bids_files[key] = {
            'subject': subject,
            'session': session,
            'task': task,
            'run': run,
            'site': site,
            'tr_value': tr_value
        }

    # >>> keep only vol_idx + ROI_* columns in the aggregated data
    keep_cols = ['vol_idx'] + [c for c in tsv_df.columns if c.startswith('ROI_')]
    chunk = tsv_df[keep_cols]

    # Concatenate rows (timepoints) for each key
    if 'data' in tsv_bids_files[key]:
        tsv_bids_files[key]['data'] = pd.concat(
            [tsv_bids_files[key]['data'], chunk], ignore_index=True
        )
    else:
        tsv_bids_files[key]['data'] = chunk

# ─────────────────────────────────────────────────────────────────────────────
# Build the MEGA wide DataFrame with your exact columns list
# NOTE: add 'key' column (stringified tuple) to help group each run
# e.g., final order: ['key','subject','session','task','run','site','vol_idx', <labels...>]
# ─────────────────────────────────────────────────────────────────────────────
columns = ['key','subject','session','task','run','site','vol_idx'] + parcellation_labels
try:
    columns
except NameError:
    columns = ['key','subject','session','task','run','site','vol_idx'] + parcellation_labels
else:
    if 'key' not in columns:
        columns = ['key'] + columns  # prepend

mega_parts = []

def _sorted_roi_cols(df):
    roi_cols = [c for c in df.columns if c.startswith("ROI_")]
    roi_cols = sorted(roi_cols, key=lambda c: int(c.split("_")[1]))
    return roi_cols

for (subject, session, task, run, site), rec in tsv_bids_files.items():
    df = rec['data'].copy()

    roi_cols = _sorted_roi_cols(df)
    if not roi_cols:
        continue

    max_roi_idx = int(roi_cols[-1].split("_")[1])
    if max_roi_idx > len(parcellation_labels):
        raise ValueError(
            f"Found {max_roi_idx} ROI columns but only {len(parcellation_labels)} labels."
        )

    # Map ROI_i -> label
    rename_map = {f"ROI_{i}": parcellation_labels[i-1] for i in range(1, max_roi_idx + 1)}
    roi_df = df[roi_cols].rename(columns=rename_map)

    # meta columns (order will be enforced by reindex later)
    key_str = f"{subject or ''}|{session or ''}|{task or ''}|{run or ''}|{site or ''}"
    roi_df['key']     = key_str
    roi_df['subject'] = subject if subject is not None else pd.NA
    roi_df['session'] = session if session is not None else pd.NA
    roi_df['task']    = task if task is not None else pd.NA
    roi_df['run']     = run if run is not None else pd.NA
    roi_df['site']    = site if site is not None else pd.NA

    # keep vol_idx
    roi_df['vol_idx'] = df['vol_idx'].astype('int64')

    # Ensure exact schema
    missing_cols = [c for c in columns if c not in roi_df.columns]
    for mc in missing_cols:
        roi_df[mc] = np.nan
    roi_df = roi_df.reindex(columns=columns)

    mega_parts.append(roi_df)

# Concatenate across all keys (all sites/subjects/sessions/runs)
if mega_parts:
    mega_df = pd.concat(mega_parts, ignore_index=True)
else:
    mega_df = pd.DataFrame(columns=columns)

# Optional: enforce dtypes for meta columns (string NA-capable)
for m in ['key','subject','session','task','run','site']:
    mega_df[m] = mega_df[m].astype('string')

# ─────────────────────────────────────────────────────────────────────────────
# Save with high-precision floats suitable for analysis
# ─────────────────────────────────────────────────────────────────────────────
out_mega_tsv = "/data/debug-proc/CPIP_mega_timeseries.tsv"
mega_df.to_csv(out_mega_tsv, sep="\t", index=False, float_format="%.17g")
print(f"[OK] Wrote mega TSV: {out_mega_tsv} with shape {mega_df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Harmonize the volumes TRs due to Calgary dummys == remove dummy number of volumes from sickids + montreal
# ─────────────────────────────────────────────────────────────────────────────
mega_df_censored = mega_df.copy()

# ensure numeric vol_idx
mega_df_censored['vol_idx'] = pd.to_numeric(mega_df_censored['vol_idx'], errors='coerce')

# target sites
targets = ['montreal', 'toronto']

# Drop vol_idx 0–7 for those sites only
drop_mask = mega_df_censored['site'].isin(targets) & mega_df_censored['vol_idx'].between(0, 7)
mega_df_censored = mega_df_censored.loc[~drop_mask].reset_index(drop=True)

# Subtract 8 from remaining rows of those sites
sub_mask = mega_df_censored['site'].isin(targets) & mega_df_censored['vol_idx'].notna()
mega_df_censored.loc[sub_mask, 'vol_idx'] = mega_df_censored.loc[sub_mask, 'vol_idx'] - 8

# keep integer dtype with NA support
mega_df_censored['vol_idx'] = mega_df_censored['vol_idx'].astype('Int64')

# save it
mega_df_censored.to_csv("/data/debug-proc/censored_mega_timeseries.tsv",sep="\t")

# ─────────────────────────────────────────────────────────────────────────────
# Harmonize the volumes TRs with high framewise displacement taking into consideration that we removed 8 volumes from montreal and sickids
# ─────────────────────────────────────────────────────────────────────────────
# mega_df_censored = pd.read_csv("/data/debug-proc/censored_mega_timeseries.tsv",sep="\t")


#get the individual runs and find the correct json files which contain the motion outliers
ind_runs = mega_df_censored.key.unique()

DUMMIES = 8  # adjust if needed

for key_id in ind_runs:
    row = mega_df_censored.loc[mega_df_censored["key"] == key_id]
    if row.empty:
        print(f"[WARN] key {key_id} not found in mega_df_censored; skipping")
        continue
    row = row.iloc[0]

    subject_id = row["subject"]
    session_val = row["session"]
    task_id    = row["task"]
    run_val    = row["run"]
    site       = row["site"]

    config     = site_configs[site]
    tedana_dir = config["tedana_dir"]
    bids_path  = config["bids_path"]

    ignore_patterns = get_ignore_patterns(bids_path)
    tedana_layout = BIDSLayout(
        tedana_dir, validate=False, derivatives=True, ignore=ignore_patterns
    )

    files = tedana_layout.get(
        subject=subject_id,
        session=session_val if not pd.isna(session_val) else None,
        task=task_id,
        run=int(run_val) if not pd.isna(run_val) else None,
        datatype="func",
        space="MNI152NLin2009cAsym",
        suffix="censoridx",
        extension="json",  # <- no leading dot
    )
    censored_volume_json = [
        f for f in files if "denoisedREGRESSEDFltcensored_" in f.filename
    ]
    if not censored_volume_json:
        print(f"[WARN] No censoridx JSON for key {key_id}; skipping")
        continue

    with open(censored_volume_json[0].path, "r") as f:
        censored_volume_data = json.load(f)

    censored_volume_ids = censored_volume_data.get("motion_outlier_idx")
    if not censored_volume_ids:
        # could be None or empty list
        censored_volume_ids = []

    # shift by dummies and keep non-negative
    censored_volume_ids_adjusted = [i - DUMMIES for i in censored_volume_ids if (i - DUMMIES) >= 0]

    # Build a base mask LIMITED to the current (sub, ses, task, run)
    base_mask = mega_df_censored["subject"].eq(subject_id) & mega_df_censored["task"].eq(task_id)

    if pd.isna(session_val):
        base_mask &= mega_df_censored["session"].isna()
    else:
        base_mask &= mega_df_censored["session"].eq(session_val)

    if pd.isna(run_val):
        base_mask &= mega_df_censored["run"].isna()
    else:
        base_mask &= mega_df_censored["run"].eq(int(run_val))

    # Initialize flags for just this run to 0, then set 1 for censored TRs
    mega_df_censored.loc[base_mask, "motion_outlier_flag"] = 0
    if censored_volume_ids_adjusted:
        censored_mask = base_mask & mega_df_censored["vol_idx"].isin(censored_volume_ids_adjusted)
        mega_df_censored.loc[censored_mask, "motion_outlier_flag"] = 1

# Filter bad runs
filters = {}
filters["montreal"] = {1:{"subject":"1466", "task":["laluna","partlycloudy"]}}
filters["toronto"] = {1:{"subject":"4504", "task":["laluna","partlycloudy"]}}
filters["calgary"] = {1:{"subject":"2976", "task":["partlycloudy"]}, 2:{"subject":"3328","task":["partlycloudy"]}}

def filter_out(filters,df):
    # rows to DROP (bad)
    bad_mask = pd.Series(False, index=df.index)

    for site, site_rules in filters.items():
        for _, rule in site_rules.items():
            subj  = str(rule["subject"])            # normalize dtype
            tasks = set(rule["task"])               # faster membership
            bad_mask |= (
                df["site"].eq(site) &
                df["subject"].astype(str).eq(subj) &
                df["task"].isin(tasks)
            )

    # rows to KEEP
    keep_mask = ~bad_mask

    # apply
    return df.loc[keep_mask].copy()

clean_mega_df_censored = filter_out(filters, mega_df_censored)

clean_mega_df_censored.to_csv("/data/debug-proc/censored_mega_timeseries_motion_outliers.tsv",sep="\t", index=False)




