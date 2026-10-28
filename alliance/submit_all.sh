#!/bin/bash
# Submit the full run as a chain of jobs: prepare (CPU) -> extract (GPU) -> one job per layer.
# Each job starts only if the previous one succeeded. Run on a login node after
# setup_env.sh, prefetch.sh and a successful smoke test:
#
#   bash alliance/submit_all.sh                      # $M2V_SPLIT, layers $M2V_LAYERS
#   M2V_LAYERS=8 bash alliance/submit_all.sh         # e.g. just layer 8
#   bash alliance/submit_all.sh --from extract       # skip steps that already finished
#
# The pipeline skips finished stages, so re-submitting after a failure is safe.
set -euo pipefail
cd "$(dirname "$0")/.."
export M2V_REPO="${M2V_REPO:-$PWD}"
source alliance/config.sh

from=prepare
[ "${1:-}" = "--from" ] && from="${2:?--from prepare|extract|layers}"

logs="$M2V_SCRATCH/logs"
mkdir -p "$logs"
common=(--parsable --export=ALL "${M2V_MAIL_ARGS[@]}")
gpu=(--gpus-per-node="$M2V_GPUS")
dep=()

if [ "$from" = prepare ]; then
    jid=$(sbatch "${common[@]}" --output="$logs/%x-%j.out" alliance/jobs/prepare.sbatch)
    echo "prepare: job $jid"
    dep=(--dependency=afterok:"$jid")
fi
if [ "$from" = prepare ] || [ "$from" = extract ]; then
    jid=$(sbatch "${common[@]}" "${gpu[@]}" "${dep[@]}" --output="$logs/%x-%j.out" alliance/jobs/extract.sbatch)
    echo "extract: job $jid"
    dep=(--dependency=afterok:"$jid")
fi
# layer jobs: one email per layer when it finishes or fails (not 12 "started" emails)
mail=()
[ -n "$M2V_EMAIL" ] && mail=("--mail-type=END,FAIL,TIME_LIMIT_80,ARRAY_TASKS")
jid=$(sbatch "${common[@]}" "${gpu[@]}" "${dep[@]}" "${mail[@]}" --array="$M2V_LAYERS%$M2V_MAX_PARALLEL" \
      --output="$logs/%x-%A_%a.out" alliance/jobs/layer.sbatch)
echo "layers $M2V_LAYERS: array job $jid"
if [ -n "$M2V_EMAIL" ]; then echo "Email notifications: $M2V_EMAIL"
else echo "No email notifications (set M2V_EMAIL in alliance/local.sh to get them)"; fi
echo "Logs: $logs   Progress: bash alliance/status.sh   Queue: squeue -u ${USER:-$(whoami)}"
