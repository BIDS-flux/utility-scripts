#!/usr/bin/env python3
"""Clean up relevant repositories and repository data before repushing dicom images.
inputs: 
    project_name, s3 bucket name,dicom repo name, bids branch, derivatives branch, gitlab_token, gitlab_user, gitlab_url
actions: 
    if dicom repo exists, remove the dicom repository, s3 objects in s3 bucket
    if bids branch exists, clean images from special remotes with datalad, update changes, and delete the branch
    for derivatives branch, if derivative branch exists, clean images from special remotes with datalad, update changes, and delete the branch
"""
import os
import sys
import subprocess
import gitlab
from minio import Minio
from minio.deleteobjects import DeleteObject
import argparse
import logging
from datalad import api as dl
from tempfile import TemporaryDirectory

logging.basicConfig(level=logging.INFO)

# Read environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

def get_gitlab_project(gl, project_name):
    """Get GitLab project by name."""
    try:
        project = gl.projects.get(project_name)
        logging.debug(f"Found project: {project.name}")
        return project
    except gitlab.exceptions.GitlabGetError as e:
        logging.error(f"Project '{project_name}' not found: {e}")
        raise

def drop_annex_data(gitlab_url, group_name, repository_name, branch, remotes_to_clean):
        # Clean up BIDS data
    with TemporaryDirectory() as temp_dir:
        logging.info(f"Cloning bids repo to temp dir: {temp_dir}")
        subprocess.run(
            ["datalad", "clone", f"{gitlab_url}/{group_name}/{repository_name}.git", temp_dir],cwd=temp_dir ,check=True
        )

        logging.info(f"Checkout branch '{branch}'")
        subprocess.run(["git", "checkout", branch], cwd=temp_dir, check=True)

        logging.info(f"Fetching branch '{branch}'")
        subprocess.run(["git", "fetch", "origin", branch], cwd=temp_dir, check=True)

        logging.info("Fetching git-annex")
        subprocess.run(["git", "fetch", "origin", "git-annex"], cwd=temp_dir, check=True)

        logging.info("Initializing git-annex")
        subprocess.run(["git", "annex", "init"], cwd=temp_dir, check=True)

        logging.info(f"Remove the content from the remotes {remotes_to_clean}")
        for remote in remotes_to_clean:
            logging.info(f"Dropping content from remote: {remote}")
            subprocess.run(["git", "annex", "drop", f"--from={remote}", "--force"], cwd=temp_dir, check=True)

        push_remotes = ["origin"] + remotes_to_clean + ["origin"]
        logging.info(f"Push the changes to the {push_remotes} in that order")
        for remote in push_remotes:
            logging.info(f"Pushing to remote: {remote}")
            subprocess.run(["datalad", "push", "--to", remote, "--force", "all"], cwd=temp_dir, check=True)

def clean_b4_push(gitlab_url: str,
                      gitlab_token: str,
                      group_name: str,
                      minio_url: str,
                      s3_bucket_name: str,
                      s3_object_prefix: str,
                      dicom_repo_name: str,
                      bids_branch: str,
                      remotes_to_clean: list,
                      derivatives_branch: list) -> None:
    """Clean up repository before repushing changes images."""

    logging.info("Starting cleanup process...")
    # initialize GitLab client
    try:
        gl = gitlab.Gitlab(gitlab_url, private_token=gitlab_token)
        gl.auth()
        logging.debug("Authenticated with GitLab successfully.")
    except Exception as e:
        logging.error(f"Failed to authenticate with GitLab: {e}")
        raise

    # Get the dicom project
    dicom_project = get_gitlab_project(gl, f"{group_name}/{dicom_repo_name}")

    if dicom_project:
        logging.info(f"Removing DICOM repository: {dicom_repo_name}")
        try:
            dicom_project.delete()
            logging.info("DICOM repository deleted successfully.")
        except Exception as e:
            logging.error(f"Failed to delete DICOM repository: {e}")
            raise
    
    # Get the S3 objects from bucket and remove them  
    if not minio_url or not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        logging.error("MinIO URL or AWS credentials are not set.")
        raise ValueError("MinIO URL or AWS credentials are not set.")
    try:
        minio_client = Minio(minio_url, access_key=AWS_ACCESS_KEY_ID, secret_key=AWS_SECRET_ACCESS_KEY)
        if minio_client.bucket_exists(s3_bucket_name):
            logging.info(f"Removing objects from S3 bucket: {s3_bucket_name}")
            # list the objects in the bucket with the specified prefix
            delete_object_list = map(
                                    lambda x: DeleteObject(x.object_name), 
                                    minio_client.list_objects(s3_bucket_name,
                                                               prefix=s3_object_prefix, 
                                                               recursive=True)
                                    )
            # remove the objects from the bucket
            errors = list(minio_client.remove_objects(delete_object_list))
            # check for erros and raise an exception if any errors occurred
            if errors:
                for error in errors:
                    raise Exception(f"ERROR occured when trying to delete object, {error}")
            logging.info("S3 bucket deleted successfully.")
        else:
            logging.warning(f"S3 bucket '{s3_bucket_name}' does not exist.")
    except Exception as e:
        logging.error(f"Failed to access S3 bucket: {e}")
        raise

    # Clean up BIDS data
    try:
        logging.info(f"Cleaning up BIDS data in branch: {bids_branch}")
        drop_annex_data(gitlab_url, group_name, "bids", bids_branch, remotes_to_clean)
    except Exception as e:
        logging.error(f"Failed to drop annex data in BIDS repository: {e}")
        raise

    # Delete the BIDS branch from the GitLab project
    try:
        bids_project = get_gitlab_project(gl, f"{group_name}/bids")
        if bids_branch in [branch.name for branch in bids_project.branches.list()]:
            logging.info(f"Deleting BIDS branch: {bids_branch}")
            bids_project.branches.delete(bids_branch)
            logging.info("BIDS branch deleted successfully.")
        else:
            logging.warning(f"BIDS branch '{bids_branch}' does not exist.")
    except Exception as e:
        logging.error(f"Failed to delete BIDS branch: {e}")
        raise

    # if derivatives_branch is not none then clean up the derivatives data
    if derivatives_branch:
        for derivative in derivatives_branch:

            # Parse the derivatives repository name
            derivative_repo_name = derivative.split('/')[0] if '/' in derivative else derivative
            full_derivative_name = f"derivatives/{derivative_repo_name}"

            # make a special case for mriqc
            if derivative_repo_name == "mriqc":
                full_derivative_name = "qc/mriqc"

            # drop annex data for the derivatives branch
            try:
                logging.info(f"Cleaning up derivative data in branch: {derivatives_branch}")
                drop_annex_data(gitlab_url, group_name, full_derivative_name, derivative, ["deriv-store"])
            except Exception as e:
                logging.error(f"Failed to drop annex data for derivatives branch '{derivative}': {e}")
                raise

            # delete the derivatives branch from the GitLab project
            try:
                derivatives_project = get_gitlab_project(gl, f"{group_name}/derivatives")
                if derivative in [branch.name for branch in derivatives_project.branches.list()]:
                    logging.info(f"Deleting derivatives branch: {derivative}")
                    derivatives_project.branches.delete(derivative)
                    logging.info("Derivatives branch deleted successfully.")
                else:
                    logging.warning(f"Derivatives branch '{derivative}' does not exist.")
            except Exception as e:
                logging.error(f"Failed to delete derivatives branch: {e}")
                raise

def main():
    parser = argparse.ArgumentParser(description="Clean up repository before repushing changes images.")
    parser.add_argument("--group_name", required=True, help="Name of the project")
    parser.add_argument("--s3_bucket_name", required=True, help="Name of the S3 bucket")
    parser.add_argument("--s3_object_prefix", required=True, help="Name of the S3 object prefix: this is ussually the DICOM StudyID")
    parser.add_argument("--dicom_repo_name", required=True, help="Name of the DICOM repository")
    parser.add_argument("--bids_branch", required=True, help="Name of the BIDS branch")
    parser.add_argument("--remotes_to_clean", nargs='+', required=True, help="List of remotes to clean")
    parser.add_argument("--derivatives_branch", nargs='+', required=True, help="List of the derivative branchs to clean")
    parser.add_argument("--gitlab_token", required=True, help="GitLab access token")
    parser.add_argument("--gitlab_user", required=True, help="GitLab username")
    parser.add_argument("--gitlab_url", required=True, help="GitLab URL")
    parser.add_argument("--minio_url", required=True, help="Minio URL")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    try:
        clean_b4_push(gitlab_url=args.gitlab_url,
                      minio_url=args.minio_url,
                      gitlab_token=args.gitlab_token,
                      gitlab_user=args.gitlab_user,
                      group_name=args.group_name,
                      s3_bucket_name=args.s3_bucket_name,
                      s3_object_prefix=args.s3_object_prefix,
                      dicom_repo_name=args.dicom_repo_name,
                      bids_branch=args.bids_branch,
                      remotes_to_clean=args.remotes_to_clean,
                      derivatives_branch=args.derivatives_branch)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
