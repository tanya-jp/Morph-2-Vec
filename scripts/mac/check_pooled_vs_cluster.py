"""
Quick check (~2 min, CPU) that pooled word-token vectors line up with the cluster's analogy spec:
3CosAdd on the first `--trials` `basic` trials of the cluster's raw-wav2vec results for that layer
(outputs/analogy/runs_id/<dataset>/w2v2_<L>/experiment_results.csv), using up to 50 random instances
per word instead of the cluster's exact samples. The rank is the position of the first token of the
correct word among all tokens (the base word excluded).

Aligned vectors give ranks close to the cluster's (mostly 0-10). Misaligned ones (e.g. extracted
in a different utterance order) give ranks in the thousands. Run from the repo root (morph2vec env):

    python scripts/mac/check_pooled_vs_cluster.py --layer 8
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.analysis.analogy import analogy_distances  # noqa: E402
from src.analysis.state_space import StateSpaceAnalysisSpec  # noqa: E402


def main(args):
    dataset = "librispeech-train-clean-100"
    d = REPO_ROOT / "outputs/pooled_states" / dataset
    agg = torch.from_numpy(np.load(d / f"w2v2_{args.layer}.npy"))
    tok_label = np.load(d / "agg_src.npy")[:, 0]
    spec = StateSpaceAnalysisSpec.from_hdf5(
        str(REPO_ROOT / "outputs/analogy/inputs" / dataset / "w2v2/state_space_spec.h5"))
    label_idx = {label: i for i, label in enumerate(spec.labels)}
    trials = pd.read_csv(REPO_ROOT / "outputs/analogy/runs_id" / dataset / f"w2v2_{args.layer}/experiment_results.csv",
                         low_memory=False)
    trials = trials[trials.experiment == "basic"].head(args.trials)
    rng = np.random.default_rng(0)

    def tokens(word, k=50):
        t = np.where(tok_label == label_idx[word])[0]
        return rng.choice(t, min(k, len(t)), replace=False)

    rows = []
    for _, r in trials.iterrows():
        a, a_star = eval(r.from_equiv_label)
        b, b_star = eval(r.to_equiv_label)
        A, A_star, B = tokens(a), tokens(a_star), tokens(b)
        n = min(len(A), len(A_star), len(B))
        dist = analogy_distances(agg, torch.tensor(A[:n]), torch.tensor(A_star[:n]), torch.tensor(B[:n]), agg)
        dist[torch.from_numpy(tok_label == label_idx[b])] = np.inf
        ranked = tok_label[dist.argsort().numpy()]
        rows.append((f"{a}:{a_star}::{b}:{b_star}", r.gt_label_rank, int(np.where(ranked == label_idx[b_star])[0][0])))
    df = pd.DataFrame(rows, columns=["analogy", "cluster_rank", "pooled_rank"])
    print(df.to_string(index=False))
    med = df.pooled_rank.median()
    print(f"\nmedian rank: cluster {df.cluster_rank.median():.0f}, pooled {med:.0f}")
    if med > 20:
        sys.exit("FAIL: pooled vectors do not line up with the spec (ranks far above the cluster's)")
    print("PASS: pooled vectors line up with the spec")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--trials", type=int, default=30)
    main(p.parse_args())
