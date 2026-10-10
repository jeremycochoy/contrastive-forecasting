#!/bin/bash
# freq_family — the code folder that a script of this folder reads, in CODE.
#
# Source it with HERE (the folder of the script) and BASE (the base folder
# of the waves) set. CODE is FF_CODE when the caller gives it. If not, it is
# the deployed folder that holds the script: deploy.sh writes
# DEPLOYED_COMMIT there. For a script in a checkout, it is <base>/code.
#
# So a script of a second code folder (`code_v2`) starts no code of the
# first, and no wave mixes the code of two deploys.
CODE="$(cd "$HERE/../../.." && pwd)"
[ -f "$CODE/DEPLOYED_COMMIT" ] || CODE="$BASE/code"
CODE="${FF_CODE:-$CODE}"
