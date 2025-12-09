#!/bin/bash
# SCRIPT will remove the annexed contents from the special remote and push the changes to origin
# this is uselful when trying to remove branches and their annexed data
export BRANCH=${1}
git checkout $BRANCH
git annex init
git annex drop --from=ACH.bids-store * --force || true
git annex drop --from=ACH.deriv-store * --force || true
git annex drop --from=ACH.bids-store.sensitive * --force || true
datalad push --to origin --force gitpush
datalad push --to ACH.bids-store --force all || true
datalad push --to ACH.deriv-store --force all || true
datalad push --to ACH.bids-store.sensitive --force all || true
datalad push --to origin --force gitpush
# git annex sync --content
git checkout main
git branch -D $BRANCH
