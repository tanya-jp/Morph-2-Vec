#!/bin/bash
# One-time setup on an Alliance login node: Python environment + scratch folders.
#
#   cd ~/Morph-2-Vec && bash alliance/setup_env.sh
#
# Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
export M2V_REPO="${M2V_REPO:-$PWD}"
source alliance/config.sh
m2v_modules

echo "== virtual environment: $M2V_VENV"
if [ ! -d "$M2V_VENV" ]; then
    mkdir -p "$(dirname "$M2V_VENV")"
    virtualenv --no-download "$M2V_VENV"
fi
source "$M2V_VENV/bin/activate"
pip install --no-index --upgrade pip

echo "== PyTorch from the Alliance wheelhouse (CUDA build)"
pip install --no-index torch torchaudio

echo "== project requirements"
# pip prefers Alliance's wheelhouse and falls back to PyPI (login nodes have internet).
# pyarrow is provided by the arrow module loaded above.
pip install -r requirements.txt

echo "== spaCy model for POS counts"
python -c "import en_core_web_trf" 2>/dev/null || python -m spacy download en_core_web_trf

echo "== scratch folders: $M2V_SCRATCH"
mkdir -p "$M2V_SCRATCH/logs" "$HF_HOME"
# Point the big pipeline folders at /scratch so the repo in $HOME stays small.
# data/ is linked as a whole. outputs/ stays a real folder because git tracks
# outputs/feasibility/; each pipeline output folder inside it is linked instead.
link_to_scratch() {  # $1 = path inside the repo
    local target="$M2V_SCRATCH/$1"
    mkdir -p "$target"
    if [ -L "$1" ]; then
        return
    elif [ -e "$1" ]; then
        echo "WARNING: $M2V_REPO/$1 already exists and is not a link; move its contents to $target and re-run." >&2
    else
        ln -s "$target" "$1"
        echo "linked $1 -> $target"
    fi
}
link_to_scratch data
mkdir -p outputs
for d in preprocessed_data preprocessing hidden_states state_space_specs equivalence_datasets \
         pos_counts analogy models model_embeddings tmp logs; do
    link_to_scratch "outputs/$d"
done

echo "== check"
python - <<'EOF'
import torch, pyarrow, datasets, transformers, spacy, h5py, papermill
print("torch", torch.__version__, "| CUDA build:", torch.version.cuda)
print("pyarrow", pyarrow.__version__, "| datasets", datasets.__version__,
      "| transformers", transformers.__version__, "| spacy", spacy.__version__)
spacy.load("en_core_web_trf")
print("spaCy en_core_web_trf loads OK")
EOF
echo "Done. Next: bash alliance/prefetch.sh"
