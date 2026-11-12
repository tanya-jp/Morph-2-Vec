"""
3CosMul analogies on raw wav2vec2 hidden states ("Wav2Vec" / runs_id), one layer at a time, on a Mac.

The Mac counterpart of `M2V_ANALOGY_METHOD=3cosmul bash alliance/submit_all.sh --from layers`
for the raw model only (the word probes are trained on the cluster). For each layer it

    1. extracts the frame-level hidden states on MPS, running the transformer only up to that
       layer (identical states; see truncate_encoder in scripts/extract_hidden_states.py)
                                                       -> outputs/hidden_states/w2v2_<L>/<dataset>.h5
    2. runs notebooks/analogy/run.ipynb with 3CosMul   -> outputs/analogy/runs_id_3cosmul/<dataset>/w2v2_<L>/
    3. deletes that layer's .h5 (~55 GB on train-clean-100), unless --keep-h5

Same paths, notebook and parameters as scripts/run_pipeline.py, so scripts/layers/compare_layers.py
reads the results unchanged. Every step resumes: finished layers are skipped, a half-written .h5 is
redone, and the analogy step continues from partial_results/.

With --pooled it skips steps 1 and 3 and reads the word-token means saved by
scripts/mac/extract_pooled_states.py (one pass for all layers, ~1 GB per layer) instead:

    caffeinate -i python scripts/mac/extract_pooled_states.py --layers 0-8
    caffeinate -i python scripts/mac/run_layers_3cosmul.py --pooled --layers 0-8 --experiments comparison

With --probe it scores the word probe ("Word" / runs/) instead of the raw model. The probe is
trained and its per-frame embeddings computed on the cluster (in the cluster's utterance order, the
same as the analogy spec, so no reordering is needed). Copy each layer's embeddings to the same path
here first (~2.3 GB per layer on train-clean-100):

    outputs/model_embeddings/<dataset>/w2v2_<L>/ffff_32/word_broad_10frames_fixedlen25/<dataset>.npy

    caffeinate -i python scripts/mac/run_layers_3cosmul.py --probe --layers 6-8 --experiments comparison

Needs the split preprocessed and the analogy inputs in place:

    python scripts/run_pipeline.py --split train-clean-100 --stage fetch,download,preprocess

Run from the repo root in the morph2vec env, e.g. for layer 8 and the experiments that
compare_layers.py reads:

    export PYTORCH_ENABLE_MPS_FALLBACK=1
    caffeinate -i python scripts/mac/run_layers_3cosmul.py --layers 8 --experiments comparison
    python scripts/layers/compare_layers.py --analogy-method 3cosmul --models Wav2Vec --layers 8
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_pipeline as P  # noqa: E402

METHOD = "3cosmul"
# the experiments scripts/layers/compare_layers.py uses (past tense, plural / 3sg -s, random control)
COMPARISON_EXPERIMENTS = ["basic", "regular", "random_to_NNS", "random_to_VBZ"]


def extract(paths: P.Paths, split: str, layer: int, device: str):
    h5 = paths.hidden_states(split, layer)
    if P.is_done(h5):
        P.log(f"extract layer {layer}: up to date ({h5})")
        return
    if h5.exists():
        P.log(f"extract layer {layer}: removing incomplete {h5}")
        h5.unlink()
    P.run_python([
        "scripts/extract_hidden_states.py",
        f"hydra.run.dir={P.hydra_str(Path('outputs/hidden_states') / f'_logs_{paths.dataset(split)}')}",
        f"base_model={paths.base_model(layer)}",
        f"+base_model.hidden_state_path={P.hydra_str(h5)}",
        "+base_model.truncate_above_extracted=true",
        f"dataset.processed_data_dir={P.hydra_str(paths.preprocessed(split))}",
        f"device={device}",
    ])
    P.mark_done(h5)


def pooled_path(paths: P.Paths, split: str, layer: int) -> Path:
    return Path("outputs/pooled_states") / paths.dataset(split) / f"w2v2_{layer}.npy"


def result_dir(paths: P.Paths, split: str, layer: int, probe: str | None) -> Path:
    if probe:
        return paths.analogy_run(split, split, layer, probe, METHOD)
    return paths.analogy_run_id(split, layer, METHOD)


def analogy(paths: P.Paths, split: str, layer: int, device: str, experiments: list[str] | None,
            pooled: bool = False, probe: str | None = None):
    outdir = result_dir(paths, split, layer, probe)
    embeddings = str(paths.embeddings(split, split, layer, probe)) if probe else "ID"
    params = P.analogy_parameters(paths, split, outdir, paths.hidden_states(split, layer), embeddings, METHOD)
    params.update(device=device, experiments_subset=experiments,
                  pooled_states_path=str(pooled_path(paths, split, layer)) if pooled else None)
    P.run_stage(f"analogy [{METHOD}] {paths.base_model(layer)} {probe or 'identity'} on {split}",
                outdir / "experiment_results.csv",
                lambda: P.run_notebook("notebooks/analogy/run.ipynb", outdir / "run.ipynb", params))
    # say which experiments the results file holds (the cluster's holds all of them)
    (outdir / "EXPERIMENTS.txt").write_text(
        "all\n" if experiments is None else "subset (Mac run):\n" + "\n".join(experiments) + "\n")


def main(args):
    os.chdir(REPO_ROOT)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    P.set_tmp_dir(Path(args.tmp_dir).resolve())

    paths = P.Paths("w2v2", hot_dir=None)
    for needed in [paths.preprocessed(args.split), paths.analogy_inputs(args.split)]:
        if not needed.exists():
            sys.exit(f"missing {needed}: preprocess the split first (see the docstring)")
    experiments = {"all": None, "comparison": COMPARISON_EXPERIMENTS}.get(args.experiments)
    if args.experiments not in ("all", "comparison"):
        experiments = args.experiments.split(",")

    if args.probe and args.pooled:
        sys.exit("--probe and --pooled cannot be combined (the probe uses its own embeddings)")
    for layer in P.parse_layers(args.layers):
        result = result_dir(paths, args.split, layer, args.probe) / "experiment_results.csv"
        if not P.is_done(result):
            if args.probe:
                emb = paths.embeddings(args.split, args.split, layer, args.probe)
                if not emb.exists():
                    sys.exit(f"missing {emb}: copy it from the cluster first (see the docstring)")
            elif args.pooled:
                if not P.is_done(pooled_path(paths, args.split, layer)):
                    sys.exit(f"missing {pooled_path(paths, args.split, layer)}: "
                             "run scripts/mac/extract_pooled_states.py first")
            else:
                extract(paths, args.split, layer, args.device)
            analogy(paths, args.split, layer, args.analogy_device, experiments, args.pooled, args.probe)
        else:
            P.log(f"layer {layer}: {result} already done")
        h5 = paths.hidden_states(args.split, layer)
        if not args.pooled and not args.probe and not args.keep_h5 and h5.exists():
            P.log(f"removing {h5}")
            h5.unlink()
            Path(str(h5) + ".done").unlink(missing_ok=True)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    p.add_argument("--split", default="train-clean-100")
    p.add_argument("--layers", default="8", help="e.g. '8', '0-11', '6,8,10'")
    p.add_argument("--experiments", default="comparison",
                   help="'comparison' (the four compare_layers.py uses: " + ", ".join(COMPARISON_EXPERIMENTS)
                        + "), 'all' (the full notebook, ~7x longer), or a comma-separated list")
    p.add_argument("--device", default="mps", help="torch device for extraction")
    p.add_argument("--analogy-device", default="mps", help="torch device for the 3CosMul scoring")
    p.add_argument("--pooled", action="store_true",
                   help="score the word-token means from scripts/mac/extract_pooled_states.py "
                        "(no frame-level extraction)")
    p.add_argument("--probe", nargs="?", const="ffff_32", default=None,
                   help="score the word probe (default config ffff_32, as on the cluster) from its "
                        "embeddings copied from the cluster, instead of the raw model")
    p.add_argument("--keep-h5", action="store_true", help="keep each layer's hidden-state .h5")
    p.add_argument("--tmp-dir", default=str(REPO_ROOT / "outputs" / "tmp"))
    main(p.parse_args())
