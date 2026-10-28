#!/bin/bash
# Step 2 (GPU): wav2vec2 hidden states for all layers in one pass, then the
# layer-independent stages (word spans, equivalence classes, POS counts, analogy stimuli).
# Writes ~52 GB per layer of train-clean-100 to /scratch. Bart estimated ~3-4 h for the
# 12-layer extraction on his desktop; the rest took ~0.5 h.
#
#SBATCH --job-name=m2v-extract
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${M2V_REPO:-$HOME/Morph-2-Vec}/alliance/config.sh"
m2v_activate
m2v_job_env

python scripts/run_pipeline.py --split "$M2V_SPLIT" --layers "$M2V_LAYERS" \
    --stage extract,specs,equiv,pos,inputs \
    --tmp-dir "$SLURM_TMPDIR/tmp" --num-workers "$SLURM_CPUS_PER_TASK"
