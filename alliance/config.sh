# Settings for running Morph-2-Vec on Alliance Canada clusters.
#
#   source alliance/config.sh            # on a login node, from the repo root
#
# Every script in alliance/ sources this file. Override any value by exporting it
# before sourcing, e.g.  M2V_SPLIT=test-clean source alliance/config.sh

# --- edit these ------------------------------------------------------------------

# SLURM allocation (CCDB "Group Name")
export M2V_ACCOUNT="${M2V_ACCOUNT:-def-brodbeck}"

# where this repo is cloned (code only; small)
export M2V_REPO="${M2V_REPO:-$HOME/Morph-2-Vec}"

# where data/ and outputs/ live (~700 GB for all 12 layers of train-clean-100).
# NB /scratch is purged of files unused for a while: copy final results elsewhere.
export M2V_SCRATCH="${M2V_SCRATCH:-${SCRATCH:-/scratch/${USER:-$(whoami)}}/morph2vec}"

# Python virtual environment (code + libraries belong in $HOME)
export M2V_VENV="${M2V_VENV:-$HOME/venvs/morph2vec}"

# LibriSpeech split to analyze and layers to run (array indices = layer numbers)
export M2V_SPLIT="${M2V_SPLIT:-train-clean-100}"
export M2V_LAYERS="${M2V_LAYERS:-0-11}"
# at most this many layer jobs run at the same time
export M2V_MAX_PARALLEL="${M2V_MAX_PARALLEL:-6}"

# GPU request passed to sbatch --gpus-per-node. "1" = any GPU; e.g. "h100:1", or a
# MIG slice name if your cluster offers them (one wav2vec2-base layer fits easily).
export M2V_GPUS="${M2V_GPUS:-1}"

# --- derived; normally no need to edit ----------------------------------------------

# sbatch / salloc read these, so jobs are charged to the right allocation
export SBATCH_ACCOUNT="$M2V_ACCOUNT"
export SALLOC_ACCOUNT="$M2V_ACCOUNT"

export HF_HOME="$M2V_SCRATCH/hf_cache"        # wav2vec2 weights + datasets cache
export PYTHONPATH="$M2V_REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE            # HDF5 locks are unreliable on Lustre
export MPLBACKEND=Agg

m2v_modules() {
    # pyarrow comes from the arrow module on Alliance, not from pip
    module load StdEnv/2023 gcc python/3.11 arrow
}

m2v_activate() {
    m2v_modules
    # shellcheck disable=SC1091
    source "$M2V_VENV/bin/activate"
    cd "$M2V_REPO" || return 1
}

# Inside jobs: never touch the network (some clusters' compute nodes have no internet);
# everything was cached by alliance/prefetch.sh. Per-job caches go to the node-local disk.
m2v_job_env() {
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export HF_DATASETS_CACHE="$SLURM_TMPDIR/hf_datasets"
    export JUPYTER_RUNTIME_DIR="$SLURM_TMPDIR/jupyter"
    mkdir -p "$SLURM_TMPDIR/tmp" "$SLURM_TMPDIR/hot"
    echo "[m2v] job $SLURM_JOB_ID on $(hostname); split=$M2V_SPLIT; local disk:"
    df -h "$SLURM_TMPDIR" | tail -1
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
}
