#!/usr/bin/env python

import pandas as pd
import numpy as np
import argparse
import json
import os
import re
from bids import BIDSLayout
from bids.layout import parse_file_entities
import subprocess
import logging
import nibabel as nib

# Set up logging and logging level to INFO
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
lgr = logging.getLogger(__name__)
lgr.setLevel(logging.INFO)

if os.getenv('DEBUG', '0') == '1':
    lgr.setLevel(logging.DEBUG)

def get_bids_files_entities(layout,subject=None, session=None):
    """
    Retrieve metadata for all files in the BIDS directory.

    Args:
    layout (BIDSLayout): BIDSLayout object representing the BIDS directory.

    Returns:
    list: A list of dictionaries containing file metadata.
    """
    files_info = []

    # Build query dictionary
    query_args = {"return_type": "filename"}
    if subject is not None:
        query_args["subject"] = subject
    if session is not None:
        query_args["session"] = session

    # These entities are found under BIDSLayout.entities in case you need to add more for other reasons in the future 
    # Loop through files and extract desired fields
    for file in layout.get(**query_args):
        # Get the bids entities
        entities = layout.parse_file_entities(file)
        # Get the corresponding JSON sidecar file
        # json_file = layout.get_nearest(file, suffix='bold', extension='json')
        try:
            if file.endswith(".nii.gz"):
                with open(file.replace(".nii.gz",".json"), 'r') as f:
                    sidecar_metadata = json.load(f)

                phase_encoding_direction = sidecar_metadata.get('PhaseEncodingDirection', 'N/A')
                phase_encoding_polarity = sidecar_metadata.get('PhaseEncodingPolarityGE', 'N/A')
                total_readout_time = sidecar_metadata.get('TotalReadoutTime', 'N/A')
                series_number = sidecar_metadata.get('SeriesNumber', 'N/A')
            else:
                phase_encoding_direction = 'N/A'
                phase_encoding_polarity = 'N/A'
                total_readout_time = 'N/A'
                series_number = 'N/A'
        except:
            # print(f"Warning - Skiping json extraction for {file} because it likely does not have a json sidecar to acompanie it")
            phase_encoding_polarity = 'N/A'
            phase_encoding_direction = 'N/A'
            total_readout_time = 'N/A'
            series_number = 'N/A'

        file_info = {
            'filename': file,
            'subject': entities.get('subject', 'N/A'),
            'session': entities.get('session', 'N/A'),
            'suffix': entities.get('suffix', 'N/A'),
            'extension': entities.get('extension', 'N/A'),
            'datatype' : entities.get('datatype', 'N/A'),
            'acquisition': entities.get('acquisition', 'N/A'),
            'echo': entities.get('echo', 'N/A'),
            'part': entities.get('part', 'N/A'),
            'task': entities.get('task', 'N/A'),
            'dir' : get_phase_encoding(phase_encoding_direction) if entities.get('direction', 'N/A') == 'N/A' else entities.get('direction', 'N/A'),
            'series_number': series_number if entities.get('SeriesNumber', 'N/A') == 'N/A' else entities.get('SeriesNumber', 'N/A'),
            'total_readout_time': total_readout_time,
            'phase_e_direction': phase_encoding_direction,
            'phase_e_polarity': phase_encoding_polarity,
            'run': series_number if entities.get('run', 'N/A') == 'N/A' else entities.get('run', 'N/A')
        }
        
        files_info.append(file_info)
    
    return files_info

def get_phase_encoding(direction):
    # Define the phase encoding directions
    # https://github.com/rordenlab/dcm2niix/issues/674
    encodings = {
        'j': 'PA',
        'j-': 'AP',
        '-j': 'AP',
        'i': 'LR',
        '-i': 'RL',
        'i-': 'RL',
        'k': 'SI',
        '-k': 'IS',
        'k-': 'IS'
    }
    
    # Return the corresponding phase encoding, or None if not found
    return encodings.get(direction, 'Invalid direction')

def datalad_run_tedana(sub,ses,bids_entities,EchoFiles,EchoTimes,MaskFiles,TedPCA,TreeJson,Confounders,OutDir):
    lgr.debug(sub+'\n')

    if not os.path.exists(OutDir):
        os.makedirs(OutDir)

    # define the prefix for the output files
    if ses is None or ses.strip() == '':
        lgr.info(f"Running tedana for subject {sub} with entities {bids_entities} in directory {OutDir}")
        prefix = f"sub-{sub}_{bids_entities}"
    else:
        lgr.info(f"Running tedana for subject {sub} session {ses} with entities {bids_entities} in directory {OutDir}")
        prefix = f"sub-{sub}_ses-{ses}_{bids_entities}"

    # Run the command using subprocess
    cmd = [
        'tedana',
        '-e', *map(str, EchoTimes),
        '-d', *EchoFiles,
        '--out-dir', OutDir,
        '--mask', MaskFiles,
        '--prefix', prefix,
        '--fittype', "curvefit",
        '--tree', TreeJson, # provide json decission_tree file with ["^trans_.*$","^rot_.*$"] to use translation and rotation confounders
        # tree="demo_external_regressors_motion_task_models",
        '--external', Confounders,
        '--tedpca', TedPCA, # Method with which to select components in TEDPCA. PCA decomposition with the mdl, kic and aic options is based on a Moving Average (stationary Gaussian) process and are ordered from most to least aggressive. ‘kundu’ or ‘kundu-stabilize’ are selection methods that were distributed with MEICA. Users may also provide a float from 0 to 1, in which case components will be selected based on the cumulative variance explained or an integer greater than 1 in which case the specificed number of components will be selected.
        '--ica-method', "robustica",
        '--n-robust-runs', '50',
        '--verbose',
        '--overwrite'
    ]

    subprocess.run(cmd, check=True)

def main():
    # Use argparse to pass arguments identify where bids and fmriprep info is information
    parser = argparse.ArgumentParser(
        description='Give me a path to your fmriprep output and number of cores to run')
    parser.add_argument('--subject', default=None, type=str, help='Subject ID to process.')
    parser.add_argument('--session', default=None, type=str, help='Session ID to process.')
    parser.add_argument('--fmriprepDir',default=None, type=str,help="This is the full path to your fmriprep dir")
    parser.add_argument('--bidsDir',default=None, type=str,help="This is the full path to your BIDS directory")
    parser.add_argument('--tedanaDir',default=None, type=str,help="This is the full path to your TEDANA output directory")
    parser.add_argument('--tedpca', default=None, type=str, help='Number of components to select for tedpca if not provided will default to half the number of volumes')
    parser.add_argument('--tree-json',default='demo_external_regressors_single_model', type=str,help="This is the decission tree to use for tedana")

    args = parser.parse_args()
    #inputs
    subject = args.subject
    session = args.session
    prep_data = args.fmriprepDir
    bids_dir = args.bidsDir
    tedana_report = args.tedanaDir
    tedpca_provided = args.tedpca
    treejson = args.tree_json

    #check that the treejson is not an empty json file:
    if os.path.getsize(treejson) == 0:
        lgr.info("The tree json file is empty will set to default: /decision_tree.json which is a modified demo_external_regressors_single_model to contain: 'regressors': ['^trans_.*$','^rot_.*$'] which should be the motion confounders")
        treejson = '/decision_tree.json'

    # Initialize BIDS layout
    layout = BIDSLayout(bids_dir, validate=False)
    if session is None or session.strip() == '': 
        files = get_bids_files_entities(layout, subject=subject, session=None)
    else:
        files = get_bids_files_entities(layout, subject=subject, session=session)

    prep_layout =  BIDSLayout(prep_data, validate=False)
    if session is None or session.strip() == '': 
        prep_files = get_bids_files_entities(prep_layout, subject=subject, session=None)
    else:
        prep_files = get_bids_files_entities(prep_layout, subject=subject, session=session)

    # # Obtain Echo files
    #find the prefix and suffix to that echo #
    echo_files = [f for f in files if f['datatype'] == 'func' and 'echo' in f['filename'] and f['extension'] == '.nii.gz' and 'sbref' not in f['filename']]
    lgr.debug(f"Found {len(echo_files)} echo files in BIDS directory: {echo_files}")

    # echo_images=[f for root, dirs, files in os.walk(prep_data)
    #             for f in files if ('_echo-' in f) & (f.endswith('_bold.nii.gz') or f.endswith('_bold.nii'))]
    file_pattern = "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}][_rec-{reconstruction}][_run-{run}][_dir-{dir}][_part-{part}]"

    #Make a list of filenames that match the prefix
    # image_prefix_list=[re.search('(.*)_echo-',f).group(1) for f in echo_images]
    image_prefix_list = [
        os.path.basename(
            layout.build_path(
                parse_file_entities(files["filename"]),
                file_pattern,
                strict=False,
                validate=False
            )
        )
        for files in echo_files
    ]
    image_prefix_list=set(image_prefix_list)
    lgr.debug(f"Found {len(image_prefix_list)} unique image prefixes: {image_prefix_list}")

    #Make a dataframe where each column is a different argument for tedana
    data=[]
    for im_prefixes in image_prefix_list:

        #Use RegEx to find bids_entities
        bids_entities=re.search(r'task-[^_]+(?:_[a-z]+-[^_]+)*', im_prefixes).group(0)

        #Make a list of the json's w/ appropriate header info from BIDS
        if session is None or session.strip() == '': 
            ME_headerinfo=[f['filename'] for f in prep_files if f['subject'] == str(subject) and "echo" in f['filename'] and str(bids_entities) in f['filename'] and f['extension'] == ".json" and f['filename'].endswith('_desc-preproc_bold.json')]
        else:
            ME_headerinfo=[f['filename'] for f in prep_files if f['subject'] == str(subject) and f['session'] == str(session) and "echo" in f['filename'] and str(bids_entities) in f['filename'] and f['extension'] == ".json" and f['filename'].endswith('_desc-preproc_bold.json')]

        #Get the confounders
        if session is None or session.strip() == '':
            confounders=[f['filename'] for f in prep_files if f['subject'] == str(subject) and str(bids_entities) in f['filename'] and f['extension'] == ".tsv" and f['filename'].endswith('_desc-confounds_timeseries.tsv')][0]
        else:
            confounders=[f['filename'] for f in prep_files if f['subject'] == str(subject) and f['session'] == str(session) and str(bids_entities) in f['filename'] and f['extension'] == ".tsv" and f['filename'].endswith('_desc-confounds_timeseries.tsv')][0]
        # Check if the file exists
        if not os.path.exists(confounders):
            lgr.info(f"File not found: {confounders}")
        else:
            lgr.info(f"File found! Cleaning null values from confounders data...\n")

            try:
                # Load the file, treating "n/a" as NaN
                df = pd.read_csv(confounders, sep='\t', na_values=["n/a"])

                # Replace NaNs and infinite values with 0.0
                df.fillna(0.0, inplace=True)
                df.replace([np.inf, -np.inf], 0.0, inplace=True)

                # Save back to the same file
                df.to_csv(confounders, sep='\t', index=False)
                try:
                    subprocess.run(f"datalad save -r -m 'Fill infs and NaN values in {os.path.basename(confounders)}'", shell=True)
                except:
                    lgr.info("datalad not in image")
                lgr.info(f"File infs / NaN cleaned and saved back to: {confounders}")

            except Exception as e:
                lgr.info(f"Error processing the file: {e}")

        #Read Echo times out of header info and sort
        echo_times=[json.load(open(f))['EchoTime'] for f in ME_headerinfo]
        echo_times=[float(x) for x in echo_times]
        echo_times=[1000*x for x in echo_times]
        echo_times.sort()

        #Find images matching the appropriate acq prefix
        if session is None or session.strip() == '':
            acq_image_files=[f['filename'] for f in prep_files if f['subject'] == str(subject) and im_prefixes in f['filename'] and "echo" in f['filename'] and f['filename'].endswith('_desc-preproc_bold.nii.gz') and "_space-MNI152NLin2009cAsym" not in f['filename']]
        else:
            acq_image_files=[f['filename'] for f in prep_files if f['subject'] == str(subject) and f['session'] == str(session) and im_prefixes in f['filename'] and "echo" in f['filename'] and f['filename'].endswith('_desc-preproc_bold.nii.gz') and "_space-MNI152NLin2009cAsym" not in f['filename']]
        acq_image_files.sort()

        #If the tedpca is not provided, set it to half the number of volumes
        if not tedpca_provided or tedpca_provided.strip() == "":
            img = nib.load(acq_image_files[0])
            tedpca = str(int(int(img.shape[-1]) / 2))  # Half the number of volumes (timepoints)
            lgr.info(f"tedpca argument not provided, setting to half the number of volumes of the run: {tedpca} from {acq_image_files[0]}")   

        #Find masks matching the appropriate acq prefix
        print(f" processing acq {im_prefixes}")
        mask_files=[f['filename'] for f in prep_files if im_prefixes in f['filename'] and f['filename'].endswith('_desc-brain_mask.nii.gz') and "_space-MNI152NLin2009cAsym" not in f['filename']][0]

        # Load the mask file with nibabel
        mask_img = nib.load(mask_files)
        mask_data = mask_img.get_fdata()

        # Set the first bottom slice that contains info that is not 0
        # mask_data[:, :, :5] = 0
        # Example: mask_data is a 3D NumPy array (x, y, z)
        # We range from 0 to the size of the image because when read through nibabel the image is inverted
        # so down is actually up in the numpy array
        for z in range(0, mask_data.shape[2]):  # start at visual bottom
            if np.any(mask_data[:, :, z] != 0):
                mask_data[:, :, z] = 0
                break

        # Save the modified mask to a temporary file
        masked_mask_file = mask_files.replace('.nii.gz', '_bottomzero.nii.gz')
        nib.Nifti1Image(mask_data, mask_img.affine, mask_img.header).to_filename(masked_mask_file)

        # Use the new mask file for tedana
        mask_files = masked_mask_file
        
        if session is None or session.strip() == '':
            out_dir= os.path.join(f"{tedana_report}/sub-{subject}/func")
        else:
            out_dir= os.path.join(f"{tedana_report}/sub-{subject}/ses-{session}/func")

        print(prep_data,out_dir)

        data.append([subject,session,bids_entities,acq_image_files,echo_times,mask_files,tedpca,treejson,confounders,out_dir])

    InData_df=pd.DataFrame(data=data,columns=['sub','ses','bids_entities','EchoFiles','EchoTimes','MaskFiles','TedPCA','TreeJson','Confounders','OutDir'])
    args=zip(InData_df['sub'].tolist(),
            InData_df['ses'].tolist(),
            InData_df['bids_entities'].tolist(),
            InData_df['EchoFiles'].tolist(),
            InData_df['EchoTimes'].tolist(),
            InData_df['MaskFiles'].tolist(),
            InData_df['TedPCA'].tolist(),
            InData_df['TreeJson'].tolist(),
            InData_df['Confounders'].tolist(),
            InData_df['OutDir'].tolist())

    # Convert zip object to a list
    args_list = list(args)

    # Iterate through a zip object
    for acquisition in args_list:
        datalad_run_tedana(*acquisition)

if __name__ == "__main__":
    main()