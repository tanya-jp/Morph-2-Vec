"""
Print analogy results in the layout of the paper's Figs 3-5 (mean rank of the target word;
lower is better, 0 = the target word is the nearest neighbour).

Reads the `experiment_results.csv` files written by `notebooks/analogy/run.ipynb` (via
`scripts/run_pipeline.py`) for the word probe (`runs/`) and the raw base model (`runs_id/`).

    python scripts/summarize_analogy_results.py --dataset librispeech-train-clean-100 --layer 8
    python scripts/summarize_analogy_results.py --layer 8 --analogy-method 3cosmul   # runs{,_id}_3cosmul/

The false-friend analysis (Fig. 5) ports the filtering of `summary_notebooks/analogy v4.ipynb`
(cells 90-96: excluded items + "strong" false friends only), except for its inner merge with
Worldlex frequencies, which needs `data/WorldLex_Eng_US.Freq.2.txt`.
"""

from argparse import ArgumentParser
from pathlib import Path
import re

import pandas as pd

ALLOMORPHS = {"S": "s", "Z": "z", "IH Z": "ɪz"}
INFLECTION_NAMES = {"NNS": "Noun", "VBZ": "Verb"}

FF_EXCLUDE = set("a b c wreck d e eh wandering lo chiu ha hahn meek jew shew ah co des re san ol der k la ye ll".split())
FF_EXCLUDE_INFLECTED = set("bunce los oft mast hauled sward rose mind mist past".split())
STRONG_OVERRIDES = {
    ("for", "force", "S"): False, ("gray", "grace", "S"): False, ("home", "holmes", "Z"): True,
    ("how", "house", "S"): False, ("law", "loss", "S"): False, ("my", "mice", "S"): False,
    ("neck", "next", "S"): True, ("play", "place", "S"): False, ("though", "those", "Z"): True,
    ("the", "thus", "S"): False, ("you", "use", "S"): False, ("you", "use", "Z"): True,
}


def unambiguous_results(df: pd.DataFrame) -> pd.DataFrame:
    u = df[df.experiment.str.startswith("unambiguous-")].copy()
    m = u.experiment.str.extract(r"unambiguous-(NNS|VBZ)_(.+?)_to_(NNS|VBZ)_(.+)$")
    u["src_inflection"], u["src_allomorph"] = m[0], m[1]
    u["tgt_inflection"], u["tgt_allomorph"] = m[2], m[3]
    return u


def fig3(df: pd.DataFrame) -> pd.DataFrame:
    """Unambiguous noun/verb inflections, pooled over allomorphs."""
    u = unambiguous_results(df)
    return u.pivot_table(index="src_inflection", columns="tgt_inflection",
                         values="gt_label_rank", aggfunc="mean").round(1)


def fig4(df: pd.DataFrame) -> pd.DataFrame:
    """Unambiguous inflections split by the allomorph of the source and target pairs."""
    u = unambiguous_results(df)
    u["src"] = u.src_inflection.map(INFLECTION_NAMES) + " [" + u.src_allomorph.map(ALLOMORPHS) + "]"
    u["tgt"] = u.tgt_inflection.map(INFLECTION_NAMES) + " [" + u.tgt_allomorph.map(ALLOMORPHS) + "]"
    return u.pivot_table(index="src", columns="tgt", values="gt_label_rank", aggfunc="mean").round(2)


def false_friend_results(df: pd.DataFrame, false_friends_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for expt, expt_df in df[df.experiment.str.contains("FF")].groupby("experiment"):
        if min(expt_df.base_from.nunique(), expt_df.base_to.nunique()) < 10:
            continue
        expt_df = expt_df.copy()
        if expt.count("-FF-") == 2:
            allomorph_from, allomorph_to = re.findall(r"-FF-([\w\s]+)-to-.+FF-([\w\s]+)", expt)[0]
        elif re.findall(r"_([\w\s]+)-to-.+FF-([\w\s]+)", expt):
            allomorph_from, allomorph_to = re.findall(r"_([\w\s]+)-to-.+FF-([\w\s]+)", expt)[0]
        else:
            allomorph_from, allomorph_to = re.findall(r".+FF-([\w\s]+)-to-.+_([\w\s]+)", expt)[0]
        expt_df["allomorph_from"], expt_df["allomorph_to"] = allomorph_from, allomorph_to
        parts.append(expt_df)

    # within-false-friend tests
    within = df[df.experiment == "false_friends"].copy()
    within["allomorph_from"] = within.inflection_from.str.extract(r"FF-(.+)$")[0]
    within["allomorph_to"] = within.inflection_to.str.extract(r"FF-(.+)$")[0]
    parts.append(within)

    res = pd.concat(parts)
    res["inflection_from"] = res.inflection_from.str.replace("-FF-.+", "-FF", regex=True)
    res["inflection_to"] = res.inflection_to.str.replace("-FF-.+", "-FF", regex=True)
    res["inflected_from"] = res.from_equiv_label.apply(lambda x: eval(x)[1])
    res["inflected_to"] = res.to_equiv_label.apply(lambda x: eval(x)[1])

    ff_from, ff_to = res.inflection_from.str.endswith("-FF"), res.inflection_to.str.endswith("-FF")
    res = res[~(ff_from & res.base_from.isin(FF_EXCLUDE)) & ~(ff_to & res.base_to.isin(FF_EXCLUDE))]
    ff_from, ff_to = res.inflection_from.str.endswith("-FF"), res.inflection_to.str.endswith("-FF")
    res = res[~(ff_from & res.inflected_from.isin(FF_EXCLUDE_INFLECTED))
              & ~(ff_to & res.inflected_to.isin(FF_EXCLUDE_INFLECTED))]

    strong = false_friends_df.set_index(["base", "inflected", "post_divergence"]).strong.to_dict()
    strong.update(STRONG_OVERRIDES)

    def is_strong(row):
        keys = []
        if row.inflection_from.endswith("-FF"):
            keys.append((row.base_from, row.inflected_from, row.allomorph_from))
        if row.inflection_to.endswith("-FF"):
            keys.append((row.base_to, row.inflected_to, row.allomorph_to))
        return all(strong.get(k, False) for k in keys)

    return res[res.apply(is_strong, axis=1)]


def fig5(df: pd.DataFrame, ff_results: pd.DataFrame, inflection: str) -> pd.DataFrame:
    """True inflections vs (strong, filtered) false friends of the same inflection."""
    pair = [inflection, f"{inflection}-FF"]
    sel = ff_results[ff_results.inflection_from.isin(pair) & ff_results.inflection_to.isin(pair)]
    table = sel.pivot_table(index="inflection_from", columns="inflection_to",
                            values="gt_label_rank", aggfunc="mean")
    table.loc[inflection, inflection] = fig3(df).loc[inflection, inflection]
    return table.round(1).rename_axis(index="source", columns="target")


def main(args):
    root = Path(args.root)
    dataset, train_dataset = args.dataset, args.train_dataset or args.dataset
    base_model = f"{args.base_model_class}_{args.layer}"
    # 3cosadd keeps the paper's runs/ and runs_id/; other methods add a suffix (see run_pipeline.py)
    suffix = "" if args.analogy_method == "3cosadd" else f"_{args.analogy_method}"
    runs = {
        "Word probe": root / f"outputs/analogy/runs{suffix}" / train_dataset / base_model / args.model
        / "word_broad_10frames_fixedlen25" / dataset / "experiment_results.csv",
        "Base model (raw hidden states)": root / f"outputs/analogy/runs_id{suffix}" / dataset / base_model / "experiment_results.csv",
    }
    false_friends_df = pd.read_csv(root / "outputs/analogy/inputs" / dataset / args.base_model_class / "false_friends.csv")

    for name, path in runs.items():
        if not path.exists():
            print(f"[missing] {name}: {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        print(f"\n==================== {name}: {dataset}, {base_model} [{args.analogy_method}] ====================")
        print("\nFig. 3: mean rank, unambiguous NNS/VBZ (source rows -> target columns)")
        print(fig3(df).to_string())
        print("\nFig. 4: mean rank by allomorph")
        print(fig4(df).to_string())
        ff_results = false_friend_results(df, false_friends_df)
        for inflection in ("NNS", "VBZ"):
            print(f"\nFig. 5: {inflection} vs strong false friends")
            print(fig5(df, ff_results, inflection).to_string())
        random_ctrl = df[df.experiment.str.startswith("random_to_")].groupby("experiment").gt_label_rank.mean().round(1)
        print("\nRandom-pair control (mean rank):", random_ctrl.to_dict())
        print(f"Trials: {len(df)}, experiments: {df.experiment.nunique()}")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--root", default=Path(__file__).resolve().parent.parent)
    p.add_argument("--dataset", default="librispeech-train-clean-100", help="analyzed dataset")
    p.add_argument("--train-dataset", default=None, help="dataset the probe was trained on (default: --dataset)")
    p.add_argument("--layer", default=8, type=int)
    p.add_argument("--model", default="ffff_32")
    p.add_argument("--base-model-class", default="w2v2")
    p.add_argument("--analogy-method", default="3cosadd", choices=["3cosadd", "3cosmul"])
    main(p.parse_args())
