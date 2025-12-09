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
from bids.layout import BIDSLayoutIndexer
import pandas as pd
import matplotlib.pyplot as plt
import glob
import ast
from decimal import Decimal
import subprocess
from nilearn.image import index_img
import json

# Define site configurations as a dictionary
# site_configs = {
#     "calgary": {
#         "bids_path": "/data/debug-proc/CPIP_bids",
#         "tedana_dir": "/data/debug-proc/CPIP_bids/derivatives/tedan2/",
#         "fmriprep_dir": "/data/debug-proc/CPIP_bids/derivatives/fmriprep2/",
#         "subjects": "2532,3398,3136,2976,3328,2928",
#         "ses": "1a",
#         "bids_filters": {"task": ["laluna", "partlycloudy", "rest"], "run": ["1", "2", None]}
#     },
#     "toronto": {
#         "bids_path": "/data/debug-proc/CPIP_bids_toronto",
#         "tedana_dir": "/data/debug-proc/CPIP_fMRI_July042025/CPIP_fMRI_July042025/derivatives/tedana2/",
#         "fmriprep_dir": "/data/debug-proc/CPIP_fMRI_July042025/CPIP_fMRI_July042025/derivatives/fmriprep2/",
#         "subjects": "4645,4504,4131,4097,3863",
#         "ses": "1a",
#         "bids_filters": {"task": ["laluna", "partlycloudy", "rest"], "run": ["1", "2", "3", None]}
#     },
#     "montreal": {
#         "bids_path": "/data/debug-proc/CPIP_bids_montreal/",
#         "tedana_dir": "/data/debug-proc/CPIP_bids_montreal/derivatives/tedana2/",
#         "fmriprep_dir": "/data/debug-proc/CPIP_bids_montreal/derivatives/fmriprep/",
#         "subjects": "1118,1312,1466,1760,2117,1755",
#         "ses": "1a",
#         "bids_filters": {"task": ["laluna", "partlycloudy", "rest"], "run": ["1", "2", "3", None]}
#     }
# }
site_configs = {
    "calgary": {
        "bids_path": "/home/milton.camachocamach/bids",
        "subjects": "*,",
        "ses": "1a",
        "bids_filters": {"task": ["laluna", "partlycloudy", "rest"], "run": ["1", "2", None]}
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

def get_subject_list(subjects_field, bids_path):
    """Return subject IDs, expanding "*" to all participants.tsv entries."""
    if not subjects_field:
        return []

    if "*" in subjects_field.strip():
        participants_tsv = os.path.join(bids_path, "participants.tsv")
        if not os.path.exists(participants_tsv):
            raise FileNotFoundError(f"participants.tsv not found in {bids_path}")

        participants_df = pd.read_csv(participants_tsv, sep="\t")
        if "participant_id" not in participants_df.columns:
            raise ValueError("participants.tsv is missing the 'participant_id' column")

        subjects_from_file = participants_df["participant_id"].dropna().astype(str)
        return [
            subject_id[4:].strip() if subject_id.startswith("sub-") else subject_id.strip()
            for subject_id in subjects_from_file
            if subject_id.strip()
        ]

    return [sub.strip() for sub in subjects_field.split(',') if sub.strip()]

def load_bids_layout(bids_path, ignore):
    """Return a BIDSLayout for the raw dataset honoring ignore patterns."""
    indexer = BIDSLayoutIndexer(ignore=ignore) if ignore else None
    return BIDSLayout(
        bids_path,
        validate=False,
        derivatives=False,
        indexer=indexer
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main work
# ─────────────────────────────────────────────────────────────────────────────
# 1) Expand ignore patterns + build layouts
modalities = [
    {"datatype": "func", "suffix": "bold", "extra_query": {"echo": [1, 2, 3, 4]}},
    # {"datatype": "dwi", "suffix": "dwi", "extra_query": {}},
    {"datatype": "anat", "suffix": "T1w", "extra_query": {}},
    {"datatype": "fmap", "suffix": "epi", "extra_query": {}},
]

data = {}
rows = []
for site in site_configs:
    config = site_configs[site]
    bids_path = config["bids_path"]
    tedana_dir = config["tedana_dir"]
    fmriprep_dir = config["fmriprep_dir"]
    subjects = get_subject_list(config["subjects"], bids_path)
    ses = config["ses"]
    bids_filters = config["bids_filters"]

    ignore_patterns = get_ignore_patterns(bids_path)
    bids_layout = load_bids_layout(bids_path, ignore_patterns)

    # 2) Sessions handling
    sessions = ses.split(",") if ses else [None]
    
    if not subjects:
        continue

    for sub in subjects:
        for ses_label in sessions:
            print(f"Processing subject {sub}, session {ses_label} at site {site}")
            for modality in modalities:
                query = {
                    'subject': sub,
                    'datatype': modality['datatype'],
                    'suffix': modality['suffix'],
                    'extension': '.nii.gz'
                }
                if ses_label:
                    query['session'] = ses_label
                query.update(modality['extra_query'])

                nii_files = bids_layout.get(**query)
                for nifti_file in nii_files:
                    nifti_img = nib.load(nifti_file.path)
                    temporal_dim = nifti_img.shape[-1] if len(nifti_img.shape) > 3 else 1
                    entities = {
                        'site': site,
                        'datatype':   modality['datatype'],
                        'suffix':     modality['suffix'],
                        'filename':   nifti_file.filename,
                        'volumes':    temporal_dim,
                        'task':       nifti_file.entities.get('task'),
                        'run':        nifti_file.entities.get('run'),
                        'acquisition': nifti_file.entities.get('acquisition'),
                        'trs':        nifti_file.entities.get('RepetitionTime'),
                        "protocol_name": nifti_file.entities.get("ProtocolName"),
                        "MultibandAccelerationFactor": nifti_file.entities.get("MultibandAccelerationFactor"),
                        "ParallelReductionFactorInPlane": nifti_file.entities.get("ParallelReductionFactorInPlane"),
                    }

                    rows.append(entities)
df = pd.DataFrame(rows)
df.to_csv('sanity_check_for_all_sites.csv')