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

# all calgary "subjects": "2532,3398,3136,2976,3328,2928",

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
        "bids_path": "/data/debug-proc/CPIP_fMRI_July042025/CPIP_fMRI_July042025/",
        "tedana_dir": "/data/debug-proc/CPIP_fMRI_July042025/CPIP_fMRI_July042025/derivatives/tedana/",
        "fmriprep_dir": "/data/debug-proc/CPIP_fMRI_July042025/CPIP_fMRI_July042025/derivatives/fmriprep2/",
        "subjects": "3536,3675,3504,3592,3803",
        "ses": None,
        "bids_filters": {"task": ["laluna", "partlycloudy", "rest"], "run": ["1", "2", "3", None]}
    },
    "montreal": {
        "bids_path": "/data/debug-proc/CPIP_bids_montreal/",
        "tedana_dir": "/data/debug-proc/CPIP_bids_montreal/derivatives/tedana2/",
        "fmriprep_dir": "/data/debug-proc/CPIP_bids_montreal/derivatives/fmriprep/",
        "subjects": "1118,1312,1466,1760,2117",
        "ses": "1a",
        "bids_filters": {"task": ["laluna", "partlycloudy", "rest"], "run": ["1", "2", "3", None]}
    }
}

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

nii_files = []
for site in site_configs:
    config = site_configs[site]
    bids_path = config["bids_path"]
    tedana_dir = config["tedana_dir"]
    fmriprep_dir = config["fmriprep_dir"]
    subjects = config["subjects"]
    ses = config["ses"]
    bids_filters = config["bids_filters"]

    ignore_patterns = get_ignore_patterns(bids_path)
    bids_layout = BIDSLayout(bids_path, validate=False, ignore=ignore_patterns)

    # 2) Sessions handling
    sessions = ses.split(",") if ses else [None]
    
    for sub in subjects.split(','):
        for ses in sessions:
            for file in bids_layout.get(
                **bids_filters, subject=sub, session=ses, datatype='func',
                suffix='bold', extension='.nii.gz', echo=1
            ):
                nii_files.append((file,site))
tsv_bids_files = {}

for nifti_file, site in nii_files:
    entities = {
        'task':        nifti_file.entities.get('task'),
        'run':         nifti_file.entities.get('run'),
        'acquisition': nifti_file.entities.get('acquisition'),
    }
    
    ProcolName = nifti_file.entities.get("ProtocolName")
    MultibandAccelerationFactor = nifti_file.entities.get("MultibandAccelerationFactor")
    ParallelReductionFactorInPlane = nifti_file.entities.get("ParallelReductionFactorInPlane")
    tsv_bids_files[nifti_file.filename] = {
        "filename": nifti_file.filename,
        "site": site,
        "protocol_name": ProcolName,
        "MultibandAccelerationFactor": MultibandAccelerationFactor,
        "ParallelReductionFactorInPlane": ParallelReductionFactorInPlane,
    }

df = pd.DataFrame.from_dict(tsv_bids_files, orient='index')
df.to_csv("/data/debug-proc/protocol_name_different_sites.tsv",sep="\t")
