#!/usr/bin/env bash
# find_dup_branches.sh
#
# Purpose:
#   Scan BIDS repo branches (default prefix: convert/) and list the branch names
#   that contain any tracked files whose *filenames* include a given substring
#   (default: "__dup"). Results are deduplicated and written to an output file.
#
# Usage:
#   ./find_dup_branches.sh <BIDS_REPO_PATH> <OUTPUT_FILE> [NEEDLE] [BRANCH_PREFIX]
#
# Args:
#   BIDS_REPO_PATH   Path to the Git repo working tree (your BIDS clone)
#   OUTPUT_FILE      File to write matching branch names to
#   NEEDLE           (optional) substring to search in filenames, default: __dup
#   BRANCH_PREFIX    (optional) branch prefix to scan, default: convert/
#
# Notes:
#   - Only *tracked* files are considered (via `git ls-files`).
#   - We checkout each branch locally (creating a tracking branch if needed).
#   - On macOS, the default /bin/bash (3.2) is fine; no `readarray` used.
#   - Dependencies: git, grep, sed, sort, tee
#
# Example:
#   ./find_dup_branches.sh /path/to/bids_repo /tmp/dup_branches.txt __dup convert/
#
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <BIDS_REPO_PATH> <OUTPUT_FILE> [NEEDLE] [BRANCH_PREFIX]" >&2
  exit 1
fi

bids=$1
out=$2
needle="${3:-__dup}"
prefix="${4:-convert/}"

# Basic checks
command -v git >/dev/null || { echo "git not found in PATH"; exit 1; }
for c in grep sed sort tee; do command -v "$c" >/dev/null || { echo "$c not found"; exit 1; }; done
git -C "$bids" rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "'$bids' is not a Git repo"; exit 1; }

# Remember starting ref to restore later
start_branch="$(git -C "$bids" rev-parse --abbrev-ref HEAD || echo HEAD)"
start_commit="$(git -C "$bids" rev-parse HEAD)"

cleanup() {
  # Restore original position (branch if available; else commit)
  if [[ "$start_branch" != "HEAD" ]]; then
    git -C "$bids" switch -q "$start_branch" 2>/dev/null || true
  else
    git -C "$bids" checkout -q "$start_commit" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Gather unique branch names (local + remote under prefix), strip origin/
branches="$(git -C "$bids" branch -a --format='%(refname:short)' \
  | grep -E "^(origin/)?${prefix}" \
  | sed 's#^origin/##' \
  | sort -u)"

# Scan each branch; print branch name if any tracked file contains the needle
{
  for branch in $branches; do
    # Switch to existing local branch or create a tracking branch
    git -C "$bids" switch -q "$branch" 2>/dev/null || git -C "$bids" switch -q -c "$branch" --track "origin/$branch"

    # Check tracked filenames for the substring (fixed-string match)
    if git -C "$bids" ls-files | grep -Fq "$needle"; then
      echo "$branch"
    fi
  done
} | sort -u | tee "$out"

# (Optional) Summary
count="$(wc -l < "$out" | tr -d '[:space:]')"
echo "Branches to retrigger ($count):"
cat "$out"
