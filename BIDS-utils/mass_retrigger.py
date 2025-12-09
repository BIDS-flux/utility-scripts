#!/usr/bin/env python3
"""
Retrigger convert pipelines for desired users.

It will identify the BIDS branch, and derivatives branch for the given project and clean up the data.

It will identify the DICOM commit in which the dicom data for the specified subject was pushed.

Tetriggering the conversion pipeline in the base branch including the following variables.
STUDY_DICOMS_COMMIT=<dicomcommit>
UPSTREAM_COMMIT_SHA=dicomcommit.
CI_PIPELINES_SOURCE="pipeline"

Inputs
------
    project_name, bids_branch, derivatives_branch, gitlab_token, gitlab_user, gitlab_url

Example usage:

    Define the AWS credentials in your environment variables:
    export AWS_ACCESS_KEY_ID=your_access_key_id
    export AWS_SECRET_ACCESS_KEY=your_secret_access_key

    python clean-b4-push.py --group_name SLBRAY/BRISKP --s3_bucket_name prod.slbray.briskp.dicoms --s3_object_prefix 1.2.840.113619.6.514.290074160449574515981477083574397775235 --dicom_repo_name 001BRISKP00301.2.840.113619.6.514.290074160449574515981477083574397775235 --bids_branch convert/001BRISKP0030 --remotes_to_clean bids-store bids-store.sensitive --derivatives_branch mriqc/sub-030_ses-1 --gitlab_token glpat-xxxxxxxxxx" --gitlab_user milton.camachocamach --gitlab_url https://cpip.ucalgary.ca --minio_url cpip.ucalgary.ca:9000
"""

import os
import sys
import subprocess
import argparse
import logging
from tempfile import TemporaryDirectory
from typing import List, Optional

import gitlab
import gitlab.v4.objects
from minio import Minio
from minio.deleteobjects import DeleteObject
from datalad import api as dl  # noqa: F401 (import kept to avoid changing behaviour)
import clean_b4_push as clean
from bids import BIDSLayout

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
BIDS_BASE_BRANCH = os.getenv("BIDS_BASE_BRANCH", "base")


def retrigger_convert_pipeline(project: gitlab.v4.objects.Project,
                               dicom_commit: str,
                               dry_run: bool) -> None:
    """Retrigger the convert pipeline for the specified project."""
    if dry_run:
        logging.info("Dry run mode - skipping retriggering the convert pipeline for project: %s for dicom commit: %s", project.name, dicom_commit)
        return
    logging.info("Retrigger convert pipeline for project: %s for dicom commit: %s", project.name, dicom_commit)
    project.pipelines.create({'ref':BIDS_BASE_BRANCH, 'variables': [{'key': 'STUDY_DICOMS_COMMIT', 'value': f'{dicom_commit}'}, {'key': 'UPSTREAM_COMMIT_SHA', 'value': f'{dicom_commit}'}, {'key': 'CI_PIPELINE_SOURCE', 'value': 'pipeline'}]})

def find_dicom_commit(gitlab_url: str,
                      group_name: str,
                      bids_branch: str,
                      repository_name: str,
                      ) -> str:
    """Find the DICOM commit for the specified subject."""
        # TODO: make it a funciton
    with TemporaryDirectory() as temp_dir:
        logging.info("Cloning %s to %s", repository_name, temp_dir)

        ds = dl.clone(f"{gitlab_url}/{group_name}/{repository_name}.git", temp_dir)

        logging.info("Checking out branch '%s'", bids_branch)
        ds.repo.checkout(bids_branch)

        logging.info("Fetching branch '%s' and git-annex", bids_branch)
        ds.repo.fetch(remote="origin", refspec=bids_branch)
        ds.repo.fetch(remote="origin", refspec="git-annex")
 
        logging.info("Fetching dicoms commit for '%s'", bids_branch)
        ds.get(path="sourcedata/dicoms", get_data=False)

        # Use datalad to get the commit hash for sourcedata/dicoms
        dicoms_path = "sourcedata/dicoms"
        commit_hash = [subdataset["gitshasum"] for subdataset in ds.repo.get_submodules() if subdataset["gitmodule_name"] == dicoms_path][0]

        logging.info("DICOM commit hash: %s", commit_hash)
        return commit_hash

def get_bids_entities(bids_branch: str, gitlab_url: str, group_name: str, repository_name: str) -> tuple[str, str]:
    """Get the BIDS entities from the branch name."""
    with TemporaryDirectory() as temp_dir:
        logging.info("Cloning %s to %s", repository_name, temp_dir)
        ds = dl.clone(f"{gitlab_url}/{group_name}/{repository_name}.git", temp_dir)
        ds.repo.checkout(bids_branch)

        # read the BIDS entities
        bids_ds_layout = BIDSLayout(temp_dir, validate=False, derivatives=False)
        subjects = bids_ds_layout.get_subjects()
        sessions = bids_ds_layout.get_sessions()
        if len(subjects) == 0:
            raise ValueError(f"No subjects found in BIDS dataset at branch '{bids_branch}'")
        elif len(subjects) > 1:
            raise ValueError(f"Multiple subjects found in BIDS dataset at branch '{bids_branch}': {subjects}")
        if len(sessions) == 0:
            raise ValueError(f"No sessions found in BIDS dataset at branch '{bids_branch}'")
        elif len(sessions) > 1:
            raise ValueError(f"Multiple sessions found in BIDS dataset at branch '{bids_branch}': {subjects}")
        logging.info(f"Extracted BIDS entities - Subject: {subjects[0]}, Session: {sessions[0]}")
        
        return (subjects[0], sessions[0])

def finder(gitlab_token: str,
           gitlab_url: str,
           group_name: str,
           bids_branch: str,
           remotes_to_clean: list[str] = ["bids-store", "bids-store.sensitive"],
           derivative_remotes_to_clean: list[str] = ["deriv-store"],
           skip_bids_cleanup: bool = False,
           skip_derivatives_cleanup: bool = False,
           dry_run: bool = False,
           ):
    """Find the BIDS branch, DICOM commit for the specified bids branch for the specified project."""
    logging.info("Starting cleanup process...")

    # Authenticate with GitLab
    try:
        gl = gitlab.Gitlab(gitlab_url, private_token=gitlab_token)
        gl.auth()
    except Exception as e:
        logging.error("Failed to authenticate with GitLab: %s", e)
        raise

    repository_name = "bids"
    # The bids branch will be found and dicoms commit
    dicom_commit = find_dicom_commit(gitlab_url=gitlab_url,
                      group_name=group_name,
                      bids_branch=bids_branch,
                      repository_name=repository_name,)

    #get bids entities
    sub, ses = get_bids_entities(bids_branch=bids_branch, gitlab_url=gitlab_url, group_name=group_name, repository_name=repository_name)

    # Define the derivatives branch based on the subject and session to remove
    derivatives_branch = [f"mriqc/sub-{sub}_ses-{ses}"]

    logging.info("Clean bids branch: %s and the derivatives branch: %s with dry run: %s", bids_branch, derivatives_branch, dry_run)
    clean.clean_b4_push(
        gl=gl,
        gitlab_url=gitlab_url,
        group_name=group_name,
        bids_branch=bids_branch,
        remotes_to_clean=remotes_to_clean,
        derivative_remotes_to_clean=derivative_remotes_to_clean,
        derivatives_branch=derivatives_branch,
        skip_s3_cleanup=True,
        skip_dicom_repo=True,
        skip_bids_cleanup=skip_bids_cleanup,
        skip_derivatives_cleanup=skip_derivatives_cleanup,
        dry_run=dry_run,
        )

    # Get the bids project
    bids_project = clean.get_gitlab_project(gl,f"{group_name}/{repository_name}")

    retrigger_convert_pipeline(project=bids_project,
                        dicom_commit=dicom_commit,
                        dry_run=dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean up repositories and retrigger the conversion of the dicoms images."
    )

    # Required args
    parser.add_argument("--gitlab_token", required=True)
    parser.add_argument("--gitlab_url", required=True)
    parser.add_argument("--group_name", required=False)
    parser.add_argument("--bids_branch", required=False)
    parser.add_argument("--remotes_to_clean", nargs="+", required=False, help="List of remotes to clean (space separated)", default=["792.bids-store","792.bids-store.sensitive"])
    parser.add_argument("--derivative_remotes_to_clean", nargs="+", required=False, help="List of derivative remotes to clean (space separated)", default=["792.deriv-store"])
    parser.add_argument("--skip_derivatives_cleanup", required=False, action="store_true", help="True or False for skipping the derivative clean", default=False)
    parser.add_argument("--skip_bids_cleanup", required=False, action="store_true", help="True or False for skipping the derivative clean", default=False)
    parser.add_argument("--dry_run", action="store_true", help="Perform a dry run (no deletions or changes made)")
    return parser
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def mass_retrigger():

    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.dry_run:
        logging.info("Running in dry run mode. No changes will be made.")
    else:
        confirm = input(
            "WARNING: This script will irreversibly delete repositories, branches, and S3 objects.\n"
            "Are you absolutely sure you want to proceed? Type 'YES' to continue: "
        )
        if confirm.strip() != "YES":
            logging.info("Aborted by user.")
            sys.exit(0)
    try:
        finder(
            gitlab_token=args.gitlab_token,
            gitlab_url=args.gitlab_url,
            group_name=args.group_name,
            bids_branch=args.bids_branch,
            remotes_to_clean=args.remotes_to_clean,
            derivative_remotes_to_clean=args.derivative_remotes_to_clean,
            skip_bids_cleanup=args.skip_bids_cleanup,
            skip_derivatives_cleanup=args.skip_derivatives_cleanup,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logging.error("An error occurred: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    mass_retrigger()
