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
col() { [ -e "$1.done" ] && printf '%-13s' ok || printf '%-13s' --; }
printf '\n%-7s%-13s%-13s%-13s%-13s%-13s\n' layer hidden probe embed analogy raw-analogy
for layer in $(seq 0 11); do
    m="w2v2_$layer"
    printf '%-7s' "$layer"
    col "outputs/hidden_states/$m/$ds.h5"
    col "outputs/models/$ds/$m/ffff_32/$eq"
    col "outputs/model_embeddings/$ds/$m/ffff_32/$eq/$ds.npy"
    col "outputs/analogy/runs/$ds/$m/ffff_32/$eq/$ds/experiment_results.csv"
    col "outputs/analogy/runs_id/$ds/$m/experiment_results.csv"
    echo
done
echo
command -v diskusage_report >/dev/null && diskusage_report 2>/dev/null | head -20
