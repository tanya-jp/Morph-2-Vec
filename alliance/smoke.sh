#!/bin/bash
# Submit the smoke test (whole pipeline on test-clean, layer 8, tiny probe training).
# Checks first that setup_env.sh and prefetch.sh were run.
#
#   bash alliance/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export M2V_REPO="${M2V_REPO:-$PWD}"
source alliance/config.sh

missing=""
[ -x "$M2V_VENV/bin/python" ] || missing+="  - Python environment: run  bash alliance/setup_env.sh\n"
[ -L outputs/hidden_states ] || missing+="  - outputs/ not linked to scratch: run  bash alliance/setup_env.sh\n"
[ -e data/downloads/test-clean.tar.gz ] || [ -e data/librispeech/test-clean.done ] \
    || missing+="  - test data: run  bash alliance/prefetch.sh test-clean\n"
if [ -n "$missing" ]; then
    printf "Not ready yet:\n%b" "$missing" >&2
    exit 1
fi

logs="$M2V_SCRATCH/logs"
mkdir -p "$logs"
jid=$(sbatch --parsable --export=ALL --gpus-per-node="$M2V_GPUS" "${M2V_MAIL_ARGS[@]}" \
      --output="$logs/%x-%j.out" alliance/jobs/smoke.sbatch)
echo "Smoke test submitted: job $jid (GPU: $M2V_GPUS)"
[ -n "$M2V_EMAIL" ] && echo "Emails go to $M2V_EMAIL"
echo "Queue:     squeue -u ${USER:-$(whoami)}        (PD = waiting, R = running)"
echo "Live log:  tail -f $logs/m2v-smoke-$jid.out"
echo "Result:    bash alliance/status.sh test-clean   (layer 8 all 'ok' = passed)"
