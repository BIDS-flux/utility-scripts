#!/bin/bash

# This function will rebase the source to the target branch and merge the source to the target 
rebase_and_merge() {
  local BRANCH_SOURCE="$1"
  local BRANCH_TARGET="$2"
  local COMMIT_SPECIFIED="$3"

  echo "Processing $BRANCH_SOURCE → $BRANCH_TARGET"

  # Switch to source branch and fetch updates
  git checkout "${BRANCH_SOURCE}" || return 1
  # git fetch origin "${BRANCH_SOURCE}" || return 1

  # Save current commit ID
  local CURRENT_COMMIT_ID
  CURRENT_COMMIT_ID=$(git rev-parse HEAD)

  # Rebase onto the target branch
  git rebase --merge -X theirs "${BRANCH_TARGET}" || true

  while [ -d .git/rebase-merge ]; do
    for f in $(git diff --name-only --diff-filter=U); do
      echo "Resolving $f from $CURRENT_COMMIT_ID or $COMMIT_SPECIFIED"
      git checkout "$CURRENT_COMMIT_ID" -- "$f" || git checkout "$COMMIT_SPECIFIED" -- "$f"
    done
    GIT_EDITOR=true git rebase --continue || true
  done

  # Merge rebased source into target
  git checkout "${BRANCH_TARGET}" || return 1
  GIT_EDITOR=true git merge "${BRANCH_SOURCE}"
  git checkout "${BRANCH_SOURCE}" || return 1
  git reset --hard "origin/${BRANCH_SOURCE}" || return 1
}
