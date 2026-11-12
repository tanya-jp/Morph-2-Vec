"""
One forward pass of wav2vec2 that saves, for every requested layer, one vector per word token:
the mean of the token's frames. These are exactly the vectors notebooks/analogy/run.ipynb builds
from the frame-level hidden states (prepare_state_trajectory with agg "mean" + flatten_trajectory),
in the same order, so the analogy step can read them directly:

    outputs/pooled_states/<dataset>/w2v2_<L>.npy    float32, (num_tokens, 768)    ~1 GB per layer
    outputs/pooled_states/<dataset>/agg_src.npy     int64,   (num_tokens, 3)      (label, instance, 0)

instead of ~55 GB of frames per layer. Word boundaries come from the analogy spec
(outputs/analogy/inputs/<dataset>/w2v2/state_space_spec.h5, built on the cluster). The script
checks that the extracted frames line up with it (same total number of frames). The spec's frame
positions follow the cluster's utterance order, which differs from the local one, so the utterances
are first put in the cluster order (outputs/pooled_states/<dataset>/cluster_order.npy, from
scripts/mac/recover_cluster_order.py). The frame-count check alone cannot catch a wrong order.

The transformer only runs up to the highest requested layer (identical states; see truncate_encoder
in scripts/extract_hidden_states.py). Layer 0 is the transformer input, 1-12 the transformer layers.
Not resumable within a pass (~2 h for layers 0-8 on train-clean-100 on an M4); layers already done
are skipped. Run from the repo root in the morph2vec env:

    export PYTORCH_ENABLE_MPS_FALLBACK=1
    caffeinate -i python scripts/mac/extract_pooled_states.py --layers 0-8

Then score with:  python scripts/mac/run_layers_3cosmul.py --pooled --layers 0-8
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

import run_pipeline as P  # noqa: E402


def pooled_dir(dataset: str) -> Path:
    return Path("outputs/pooled_states") / dataset


def token_spans(spec, total_num_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (starts, ends, src) of all word tokens in flatten_trajectory order (label, then instance).
    `ends` are inclusive and clipped to the last frame, as slicing in prepare_state_trajectory
    does. Tokens with no frames would be NaN there and dropped by flatten_trajectory, so they
    are dropped here too.
    """
    starts, ends, src = [], [], []
    for label_idx, spans in enumerate(spec.target_frame_spans):
        spans = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
        starts.append(spans[:, 0])
        ends.append(spans[:, 1])
        src.append(np.stack([np.full(len(spans), label_idx), np.arange(len(spans)),
                             np.zeros(len(spans), dtype=np.int64)], axis=1))
    starts, ends, src = np.concatenate(starts), np.concatenate(ends), np.concatenate(src).astype(np.int64)
    ends = np.minimum(ends, total_num_frames - 1)
    keep = ends >= starts
    if (~keep).any():
        print(f"dropping {(~keep).sum()} tokens without frames (as flatten_trajectory does)")
    return starts[keep], ends[keep], src[keep]


def main(args):
    os.chdir(REPO_ROOT)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    P.set_tmp_dir(Path(args.tmp_dir).resolve())

    import datasets
    import torch
    import transformers
    from hydra import compose, initialize_config_dir
    from extract_hidden_states import HiddenStateWriter, extract_hidden_states, truncate_encoder
    from src.analysis.state_space import StateSpaceAnalysisSpec
    from src.models.transformer import prepare_feature_extractor

    paths = P.Paths("w2v2", hot_dir=None)
    dataset_name = paths.dataset(args.split)
    outdir = pooled_dir(dataset_name)
    out = {layer: outdir / f"w2v2_{layer}.npy" for layer in P.parse_layers(args.layers)}
    layers = [layer for layer, path in out.items() if not P.is_done(path)]
    if not layers:
        print(f"all layers already done in {outdir}")
        return
    print(f"extracting layers {layers} -> {outdir}")

    spec = StateSpaceAnalysisSpec.from_hdf5(str(paths.analogy_inputs(args.split) / "state_space_spec.h5"))
    starts, ends, src = token_spans(spec, spec.total_num_frames)
    order = np.argsort(starts, kind="stable")  # tokens by onset, to find each batch's tokens
    sorted_starts = starts[order]
    max_len = int((ends - starts).max()) + 1
    num_tokens = len(starts)
    print(f"{num_tokens} word tokens, {spec.total_num_frames} frames in the spec")

    class PooledStateWriter(HiddenStateWriter):
        """
        Takes the frames extract_hidden_states produces, batch by batch, and keeps only the mean
        of each word token's frames. Global frame index = position in the stream, as in the .h5.
        """

        def __init__(self):
            outdir.mkdir(parents=True, exist_ok=True)
            self.tmp = {layer: outdir / f"w2v2_{layer}.partial.npy" for layer in layers}
            # zero-filled; each token gets (sum of its frames in a batch) / (its frame count) added
            # from every batch it overlaps
            self.agg = {layer: np.lib.format.open_memmap(self.tmp[layer], mode="w+", dtype=np.float32,
                                                         shape=(num_tokens, args.hidden_size))
                        for layer in layers}
            self.covered = np.zeros(num_tokens, dtype=np.int64)  # frames added so far, per token
            self.num_frames = 0

        def write(self, layer_states, flat_idxs, compression_ratios):
            # PyTorch's MPS allocator keeps freed blocks for reuse; with utterances of many lengths
            # its cache grew to ~28 GB on a 32 GB Mac and the run slowed to a crawl from swapping.
            # Release it every batch (the states here are already copied to the CPU).
            if args.device == "mps":
                torch.mps.empty_cache()
            lo, n = self.num_frames, len(flat_idxs)
            hi = lo + n
            # Spans are positions in the stream of all frames (as prepare_state_trajectory slices
            # the .h5), and ~5% run into the next utterance, so a token may overlap two batches.
            rows = order[np.searchsorted(sorted_starts, lo - max_len):np.searchsorted(sorted_starts, hi)]
            rows = rows[ends[rows] >= lo]
            if len(rows):
                s, e = np.maximum(starts[rows], lo) - lo, np.minimum(ends[rows], hi - 1) - lo
                count = (ends[rows] - starts[rows] + 1)[:, None]
                for layer, states in layer_states.items():
                    x = states.double().numpy()
                    cs = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
                    self.agg[layer][rows] += ((cs[e + 1] - cs[s]) / count).astype(np.float32)
                self.covered[rows] += e - s + 1
            self.num_frames = hi

        def close(self, complete: bool = True):
            if complete:
                if self.num_frames != spec.total_num_frames:
                    raise RuntimeError(
                        f"extracted {self.num_frames} frames but the analogy spec has {spec.total_num_frames}: "
                        "the preprocessed data does not match the one the spec was built from")
                incomplete = self.covered != ends - starts + 1
                assert not incomplete.any(), f"{incomplete.sum()} word tokens did not get all their frames"
                np.save(outdir / "agg_src.npy", src)
            for layer in layers:
                self.agg[layer].flush()
                del self.agg[layer]
                if complete:
                    self.tmp[layer].rename(out[layer])
                    P.mark_done(out[layer])
                else:
                    self.tmp[layer].unlink(missing_ok=True)

    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base="1.2"):
        config = compose(config_name="config.yaml", overrides=[f"base_model=w2v2_{max(layers)}"])
    feature_extractor = prepare_feature_extractor(config)
    dataset = datasets.load_from_disk(str(paths.preprocessed(args.split)))
    # The spec's frame positions follow the utterance order of the dataset it was built from (the
    # cluster's); the local copy lists the same utterances in a different (filesystem) order.
    order_path = outdir / "cluster_order.npy"
    if order_path.exists():
        order_idx = np.load(order_path)
        assert sorted(order_idx.tolist()) == list(range(len(dataset))), f"{order_path} is not a permutation"
        print(f"reordering utterances to the cluster order from {order_path}")
        dataset = dataset.select(order_idx)
    elif not args.local_order:
        sys.exit(f"missing {order_path}: run scripts/mac/recover_cluster_order.py first, or pass --local-order "
                 "if the spec was built from this local dataset")
    dataset = dataset.with_format("torch")
    model = transformers.AutoModel.from_pretrained(config.base_model.model_ref).to(args.device)
    assert model.config.hidden_size == args.hidden_size
    truncate_encoder(model, max(layers))

    writer = PooledStateWriter()
    try:
        extract_hidden_states(dataset, model, feature_extractor, writer, layers,
                              pseudo_causal=config.base_model.pseudo_causal)
    except BaseException:
        writer.close(complete=False)
        raise
    writer.close()
    print(f"done: {', '.join(str(out[layer]) for layer in layers)}")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    p.add_argument("--split", default="train-clean-100")
    p.add_argument("--layers", default="0-8", help="e.g. '0-8', '8', '0,4,8'")
    p.add_argument("--device", default="mps")
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--tmp-dir", default=str(REPO_ROOT / "outputs" / "tmp"))
    p.add_argument("--local-order", action="store_true",
                   help="use the local utterance order (only if the spec was built from this local dataset, "
                        "e.g. test-clean preprocessed here)")
    main(p.parse_args())
