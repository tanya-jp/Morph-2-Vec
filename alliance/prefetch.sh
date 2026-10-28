#!/bin/bash
# Download everything the jobs need, on a login node (compute nodes may be offline):
# LibriSpeech archives, alignments, spelling map, CMUdict, wav2vec2-base weights.
#
#   bash alliance/prefetch.sh                 # test-clean (smoke test) + $M2V_SPLIT
#   bash alliance/prefetch.sh test-clean      # only the given split(s)
#
# Downloads resume after an interruption; re-running skips what is already there.
# train-clean-100 is ~6.3 GB; run inside `tmux` or `screen` so a dropped SSH
# connection doesn't stop it.
set -euo pipefail
cd "$(dirname "$0")/.."
export M2V_REPO="${M2V_REPO:-$PWD}"
source alliance/config.sh
m2v_activate

splits=("$@")
if [ ${#splits[@]} -eq 0 ]; then
    splits=(test-clean)
    [ "$M2V_SPLIT" != test-clean ] && splits+=("$M2V_SPLIT")
fi

for split in "${splits[@]}"; do
    echo "== fetch $split"
    python scripts/run_pipeline.py --split "$split" --layers 8 --stage fetch
done

echo "== check that the jobs will find everything offline"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'EOF'
import transformers, spacy
transformers.AutoModel.from_pretrained("facebook/wav2vec2-base")
spacy.load("en_core_web_trf")
print("wav2vec2-base and en_core_web_trf load offline: OK")
EOF
ls -lh data/downloads data/external
echo "Done. Next: bash alliance/smoke.sh (test) or bash alliance/submit_all.sh (full run)"
