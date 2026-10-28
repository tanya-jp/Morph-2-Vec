#!/bin/bash
# Step 3 (GPU, one array task per layer): train the word probe, embed, and run the
# analogy experiments for the probe and for the raw wav2vec2 states.
# The array index is the layer number:  sbatch --array=0-11%6 alliance/jobs/layer.sbatch
# Bart's desktop: ~3.5 h per layer of train-clean-100.
#
# The layer's HDF5 file (~52 GB) is copied to the node-local disk ($SLURM_TMPDIR), which
# is much faster than /scratch for the random reads done by training and the analogy.
#
#SBATCH --job-name=m2v-layer
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "${M2V_REPO:-$HOME/Morph-2-Vec}/alliance/config.sh"
m2v_activate
m2v_job_env

LAYER="${SLURM_ARRAY_TASK_ID:?submit with --array=<layers>}"
python scripts/run_pipeline.py --split "$M2V_SPLIT" --layers "$LAYER" \
    --stage train,embed,analogy \
    --tmp-dir "$SLURM_TMPDIR/tmp" --hot-dir "$SLURM_TMPDIR/hot" \
    --num-workers "$SLURM_CPUS_PER_TASK"

python scripts/summarize_analogy_results.py --dataset "librispeech-$M2V_SPLIT" --layer "$LAYER" || true
echo "[m2v] layer $LAYER finished"
