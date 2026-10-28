#!/bin/bash
# Which pipeline stages have finished, per layer (reads the runner's .done markers).
#
#   bash alliance/status.sh                 # $M2V_SPLIT
#   bash alliance/status.sh test-clean
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export M2V_REPO="${M2V_REPO:-$PWD}"
source alliance/config.sh
split="${1:-$M2V_SPLIT}"
ds="librispeech-$split"
eq=word_broad_10frames_fixedlen25
mark() { [ -e "$1.done" ] && printf '  ok  ' || printf '  --  '; }

echo "split: $split"
printf 'shared: preprocessed%s  equivalence%s  pos%s  inputs%s\n' \
    "$(mark "outputs/preprocessed_data/$ds")" \
    "$(mark "outputs/equivalence_datasets/$ds/w2v2/$eq/equivalence.pkl")" \
    "$(mark "outputs/pos_counts/$ds.pkl")" \
    "$(mark "outputs/analogy/inputs/$ds/w2v2")"
printf '\nlayer  hidden  probe   embed   analogy raw-analogy\n'
for layer in $(seq 0 11); do
    m="w2v2_$layer"
    printf '%5s' "$layer"
    mark "outputs/hidden_states/$m/$ds.h5"
    mark "outputs/models/$ds/$m/ffff_32/$eq"
    mark "outputs/model_embeddings/$ds/$m/ffff_32/$eq/$ds.npy"
    mark "outputs/analogy/runs/$ds/$m/ffff_32/$eq/$ds/experiment_results.csv"
    mark "outputs/analogy/runs_id/$ds/$m/experiment_results.csv"
    echo
done
echo
command -v diskusage_report >/dev/null && diskusage_report 2>/dev/null | head -20
