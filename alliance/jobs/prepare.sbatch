#!/bin/bash
# Step 1 (CPU only): extract the archives, FLAC -> WAV, and run the preprocessing notebook.
# Bart's desktop: ~0.5 h + ~1.5 h for train-clean-100.
#
#SBATCH --job-name=m2v-prepare
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${M2V_REPO:-$HOME/Morph-2-Vec}/alliance/config.sh"
m2v_activate
m2v_job_env

python scripts/run_pipeline.py --split "$M2V_SPLIT" --layers 8 \
    --stage download,preprocess \
    --tmp-dir "$SLURM_TMPDIR/tmp" --num-workers "$SLURM_CPUS_PER_TASK"
