#!/bin/bash
# Smoke test: the whole pipeline on test-clean, layer 8, with a tiny probe training run.
# Checks the environment, offline caches and GPU before spending hours on train-clean-100.
#
#   source alliance/config.sh && sbatch alliance/jobs/smoke.sbatch
#
#SBATCH --job-name=m2v-smoke
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${M2V_REPO:-$HOME/Morph-2-Vec}/alliance/config.sh"
m2v_activate
m2v_job_env

python scripts/run_pipeline.py --split test-clean --layers 8 --smoke \
    --stage download,preprocess,extract,specs,equiv,pos,inputs,train,embed,analogy \
    --tmp-dir "$SLURM_TMPDIR/tmp" --hot-dir "$SLURM_TMPDIR/hot" \
    --num-workers "$SLURM_CPUS_PER_TASK"

python scripts/summarize_analogy_results.py --dataset librispeech-test-clean --layer 8 || true
echo "[m2v] smoke test finished"
