#!/usr/bin/env python3
"""
Clean up relevant repositories and data before repushing DICOM images.

Inputs
------
    project_name, s3 bucket name, dicom repo name, bids branch, derivatives branch,
    gitlab_token, gitlab_user, gitlab_url

Actions (now individually skippable)
------------------------------------
    • Delete DICOM repository                (--skip_dicom_repo)
    • Delete S3 / MinIO objects              (--skip_s3_cleanup)
    • Clean + delete BIDS branch             (--skip_bids_cleanup)
    • Clean + delete derivative branches     (--skip_derivatives_cleanup)

If a --skip_* flag is provided, that step is skipped.

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
import glob
import pathlib
from tempfile import TemporaryDirectory
from typing import List, Optional

import gitlab
from minio import Minio
from minio.deleteobjects import DeleteObject
from datalad import api as dl  # noqa: F401 (import kept to avoid changing behaviour)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_gitlab_project(gl: gitlab.Gitlab, project_name: str):
    """Get GitLab project by name."""
    try:
        project = gl.projects.get(project_name)
        logging.debug("Found project: %s", project.name)
        return project
    except gitlab.exceptions.GitlabGetError as e:
        logging.error("Project '%s' not found: %s", project_name, e)
        raise


def drop_annex_data(
    gitlab_url: str,
    group_name: str,
    repository_name: str,
    branch: str,
    remotes_to_clean: List[str],
    dry_run: bool = False,
):
    """Clone repo, drop annexed content from given remotes, push changes back."""
    with TemporaryDirectory() as temp_dir:
        logging.info("Cloning %s to %s", repository_name, temp_dir)
        logging.info("dry run NOT dropping annex content from remotes: %s", remotes_to_clean)
        if not dry_run:
            ds = dl.clone(f"{gitlab_url}/{group_name}/{repository_name}.git", temp_dir)

            logging.info("Checking out branch '%s'", branch)
            ds.repo.checkout(branch)

            logging.info("Fetching branch '%s' and git-annex", branch)
            ds.repo.fetch(remote="origin", refspec=branch)
            ds.repo.fetch(remote="origin", refspec="git-annex")

            logging.info("Initializing git‑annex")
            ds.repo._call_annex(args=["init"])

            logging.info("Remotes to clean: %s", remotes_to_clean)
            # get all the files in temp_dir
            for remote in remotes_to_clean:
                logging.info("Dropping annex content from branch: %s in remote: %s", branch, remote)
                # ds.repo.drop(files="*", options=[f"--branch={branch}", f"--from={remote}", "--force"])
                subprocess.run(
                    ["git", "annex", "drop", f"--from={remote}", "--force"],
                    cwd=temp_dir,
                    check=True,
                )

            # define origin first, then other remotes, then origin again
            push_order = ["origin"] + remotes_to_clean + ["origin"]
            logging.info("Pushing changes in order: %s", push_order)
            for remote in push_order:
                ds.push(to=remote, force="all")


# ---------------------------------------------------------------------------
# Core cleanup routine
# ---------------------------------------------------------------------------
def clean_b4_push(
    *,
    gl: Optional[gitlab.Gitlab] = None,
    gitlab_url: Optional[str] = None,
    gitlab_token: Optional[str] = None,
    group_name: Optional[str] = None,
    minio_url: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_object_prefix: Optional[str] = None,
    dicom_repo_name: Optional[str] = None,
    bids_branch: Optional[str] = None,
    remotes_to_clean: Optional[List[str]] = None,
    derivative_remotes_to_clean: Optional[List[str]] = None,
    derivatives_branch: Optional[List[str]] = None,
    skip_dicom_repo: bool = False,
    skip_s3_cleanup: bool = False,
    skip_bids_cleanup: bool = False,
    skip_derivatives_cleanup: bool = False,
    dry_run: bool = False,
) -> None:
    """Clean up repository before repushing images, with optional skips."""

    logging.info("Starting cleanup process...")

    # Authenticate with GitLab
    try:
        if gl is None:
            gl = gitlab.Gitlab(gitlab_url, private_token=gitlab_token)
            gl.auth()
        else:
            logging.info("Using provided GitLab instance.")
            gl.auth()
    except Exception as e:
        logging.error("Failed to authenticate with GitLab: %s", e)
        raise

    # -----------------------------------------------------------------------
    # DICOM repository
    # -----------------------------------------------------------------------
    if skip_dicom_repo:
        logging.info("Skipping DICOM repo deletion (--skip_dicom_repo)")
    else:
        dicom_project = get_gitlab_project(
            gl, f"{group_name}/sourcedata/dicoms/{dicom_repo_name}"
        )
        if dicom_project:
            if dry_run:
                logging.info("dry run NOT Deleting DICOM repository: %s", dicom_repo_name)
            if not dry_run:
                logging.info("Deleting DICOM repository: %s", dicom_repo_name)
                dicom_project.delete()

    # -----------------------------------------------------------------------
    # S3 / MinIO objects
    # -----------------------------------------------------------------------
    if skip_s3_cleanup:
        logging.info("Skipping S3 cleanup (--skip_s3_cleanup)")
    else:
        if not (minio_url and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
            raise ValueError("MinIO URL or AWS credentials are not set.")

        try:
            minio_client = Minio(
                minio_url,
                access_key=AWS_ACCESS_KEY_ID,
                secret_key=AWS_SECRET_ACCESS_KEY,
            )
            if minio_client.bucket_exists(s3_bucket_name):
                logging.info("dry run NOT Removing objects from bucket: %s", s3_bucket_name)

                delete_objs = (
                    DeleteObject(obj.object_name)
                    for obj in minio_client.list_objects(
                        s3_bucket_name, prefix=s3_object_prefix, recursive=True
                    )
                )
                if not dry_run:
                    logging.info("Removing objects from bucket: %s", s3_bucket_name)
                    errors = list(
                        minio_client.remove_objects(
                            bucket_name=s3_bucket_name, delete_object_list=delete_objs
                        )
                    )
                    if errors:
                        raise RuntimeError(
                            "Errors occurred while deleting S3 objects: %s" % errors
                        )
                    logging.info("S3 objects deleted successfully.")
            else:
                logging.warning("S3 bucket '%s' does not exist.", s3_bucket_name)
        except Exception as e:
            logging.error("Failed to access S3 bucket: %s", e)
            raise

    # -----------------------------------------------------------------------
    # BIDS branch
    # -----------------------------------------------------------------------
    if skip_bids_cleanup:
        logging.info("Skipping BIDS cleanup (--skip_bids_cleanup)")
    else:
        try:
            logging.info("Cleaning BIDS branch: %s", bids_branch)
            drop_annex_data(
                gitlab_url, group_name, "bids", bids_branch, remotes_to_clean, dry_run=dry_run
            )
        except Exception as e:
            logging.error("Failed to drop annex data in BIDS repo: %s", e)
            raise

        try:
            bids_project = get_gitlab_project(gl, f"{group_name}/bids")
            branches = [b.name for b in bids_project.branches.list(get_all=True)]
            if bids_branch in branches:
                if dry_run:
                    logging.info("dry run NOT Deleting BIDS branch: %s", bids_branch)
                if not dry_run:
                    logging.info("Deleting BIDS branch: %s", bids_branch)
                    bids_project.branches.delete(bids_branch)
            else:
                logging.warning("BIDS branch '%s' does not exist.", bids_branch)
        except Exception as e:
            logging.error("Failed to delete BIDS branch: %s", e)
            raise

    # -----------------------------------------------------------------------
    # Derivative branches
    # -----------------------------------------------------------------------
    if skip_derivatives_cleanup:
        logging.info("Skipping derivatives cleanup (--skip_derivatives_cleanup)")
    elif derivatives_branch:
        for derivative in derivatives_branch:
            derivative_repo = derivative.split("/")[0]
            repo_path = (
                "qc/mriqc"
                if derivative_repo == "mriqc"
                else f"derivatives/{derivative_repo}"
            )

            try:
                logging.info("Cleaning derivative branch: %s", derivative)
                drop_annex_data(
                    gitlab_url,
                    group_name,
                    repo_path,
                    derivative,
                    derivative_remotes_to_clean,
                    dry_run=dry_run,
                )
            except Exception as e:
                logging.error(
                    "Failed to drop annex data for derivatives branch '%s': %s",
                    derivative,
                    e,
                )
                raise

            try:
                deriv_project = get_gitlab_project(gl, f"{group_name}/{repo_path}")
                branches = [b.name for b in deriv_project.branches.list(get_all=True)]
                if derivative in branches:
                    if dry_run:
                        logging.info("dry run NOT Deleting derivative branch: %s", derivative)
                    if not dry_run:
                        logging.info("Deleting derivative branch: %s", derivative)
                        deriv_project.branches.delete(derivative)
                else:
                    logging.warning(
                        "Derivatives branch '%s' does not exist.", derivative
                    )
            except Exception as e:
                logging.error("Failed to delete derivatives branch: %s", e)
                raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Clean up repositories and data before repushing images."
    )

    # Required args
    parser.add_argument("--group_name", required=False)
    parser.add_argument("--s3_bucket_name", required=False)
    parser.add_argument("--s3_object_prefix", required=False)
    parser.add_argument("--dicom_repo_name", required=False)
    parser.add_argument("--bids_branch", required=False)
    parser.add_argument("--remotes_to_clean", nargs="+", required=False, help="List of remotes to clean (space separated)")
    parser.add_argument("--derivative_remotes_to_clean", nargs="+", required=False, help="List of derivative remotes to clean (space separated)")
    parser.add_argument("--derivatives_branch", nargs="+", required=False, help="Derivative branches to clean (space separated)")
    parser.add_argument("--gitlab_token", required=False)
    parser.add_argument("--gitlab_user", required=False)
    parser.add_argument("--gitlab_url", required=False)
    parser.add_argument("--minio_url", required=False)
    parser.add_argument("--skip_dicom_repo", action="store_true")
    parser.add_argument("--skip_s3_cleanup", action="store_true")
    parser.add_argument("--skip_bids_cleanup", action="store_true")
    parser.add_argument("--skip_derivatives_cleanup", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Perform a dry run (no deletions or changes made)")

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
        clean_b4_push(
            gitlab_url=args.gitlab_url,
            minio_url=args.minio_url,
            gitlab_token=args.gitlab_token,
            group_name=args.group_name,
            s3_bucket_name=args.s3_bucket_name,
            s3_object_prefix=args.s3_object_prefix,
            dicom_repo_name=args.dicom_repo_name,
            bids_branch=args.bids_branch,
            remotes_to_clean=args.remotes_to_clean,
            derivative_remotes_to_clean=args.derivative_remotes_to_clean,
            derivatives_branch=args.derivatives_branch,
            skip_dicom_repo=args.skip_dicom_repo,
            skip_s3_cleanup=args.skip_s3_cleanup,
            skip_bids_cleanup=args.skip_bids_cleanup,
            skip_derivatives_cleanup=args.skip_derivatives_cleanup,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logging.error("An error occurred: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
