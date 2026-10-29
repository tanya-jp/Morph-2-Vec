"""
Past tense (VBD) across all wav2vec2-base layers: Part 4 of the feasibility notebook, run on every layer.

For each layer it reads the two analogy results files the pipeline wrote,

    outputs/analogy/runs/<dataset>/w2v2_<L>/ffff_32/<equivalence>/<dataset>/experiment_results.csv   (word probe)
    outputs/analogy/runs_id/<dataset>/w2v2_<L>/experiment_results.csv                                 (raw wav2vec2)

and computes exactly what the notebook computes for layer 8 (same code from scripts/feasibility/):

    * Figure 6: regular past tense, transfer across the endings [t] / [d] / [ɪd]
    * Figure 7: regular vs. irregular past, against plural -s, 3sg -s and the random-pair control
    * per-verb mean ranks (worst / best target verbs)

Writes to outputs/feasibility/layers/ (small; meant to be committed):

    tables/layers_conditions.csv        layer x model x condition (Figure 7 conditions)
    tables/layers_allomorph_grid.csv    layer x model x source ending x target ending (Figure 6 cells)
    tables/layers_best.csv              best layer per model and condition (top-1 accuracy, mean rank)
    tables/layers_per_verb.csv.gz       layer x model x target verb (regular past)
    tables/layer8_vs_feasibility.csv    this run's layer 8 next to the feasibility demo's layer-8 tables
    figures/fig_layers_conditions.{png,pdf}
    figures/fig_layers_allomorph_top1.{png,pdf}
    figures/fig_layers_allomorph_grids_<model>.{png,pdf}

Needs only pandas, numpy, matplotlib and seaborn (no GPU, no PyTorch). Run from the repo root:

    python scripts/layers/compare_layers.py                  # layers 0-11
    python scripts/layers/compare_layers.py --layers 6-9     # a subset
"""

from argparse import ArgumentParser
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "feasibility"))
import common as C  # noqa: E402  (feasibility scripts: shared paths, constants and statistics)

# the feasibility export functions (the file name starts with a digit, so import it by path)
_spec = importlib.util.spec_from_file_location("export_compact", HERE.parent / "feasibility" / "02_export_compact_data.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

OUT = C.FEAS / "layers"
TABLES, FIGURES = OUT / "tables", OUT / "figures"
ENDING_LABEL = {"T": "[t]", "D": "[d]", "IH D": "[ɪd]"}
ENDINGS = ["[t]", "[d]", "[ɪd]"]

# Figure 7 conditions; the names match outputs/feasibility/tables/table9_regular_vs_irregular.csv
COND_PAST_REG = "past regular"
COND_PAST_REG_CLEAN = "past regular (auto-class regular only)"
COND_PAST_IRR = "past irregular"
COND_PAST_IRR_CLEAN = "past irregular (no modals, be, homographs)"
COND_NNS = "plural NNS regular"
COND_VBZ = "3sg VBZ regular"
COND_RANDOM = "random pair (→ NNS / VBZ)"
CONDITIONS = [COND_PAST_REG, COND_PAST_REG_CLEAN, COND_PAST_IRR, COND_PAST_IRR_CLEAN, COND_NNS, COND_VBZ, COND_RANDOM]


def parse_layers(spec: str) -> list[int]:
    layers = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        layers.extend(range(int(a), int(b or a) + 1))
    return sorted(set(layers))


def results_paths(layer: int) -> dict[str, Path]:
    base = f"w2v2_{layer}"
    return {
        "Word": C.ROOT / "outputs/analogy/runs" / C.DATASET / base / C.MODEL / C.EQUIVALENCE / C.DATASET / "experiment_results.csv",
        "Wav2Vec": C.ROOT / "outputs/analogy/runs_id" / C.DATASET / base / "experiment_results.csv",
    }


def load_pairs() -> pd.DataFrame:
    """
    One row per past-tense pair, with the automatic class and homophones. Read from the committed annotation
    sheet (scripts/feasibility/03_prepare_theory_materials.py), which is layer-independent, so no spec / PyTorch
    is needed here.
    """
    p = pd.read_csv(C.THEORY / "vbd_pairs_for_annotation.csv", encoding="utf-8-sig")
    return p.rename(columns={"past": "inflected", "past_sounds_like": "past_homophones"})


# --------------------------------------------------------------------------- one layer (notebook Part 4)

def layer_tables(layer: int, pairs: pd.DataFrame, n_boot: int):
    C.RESULTS = results_paths(layer)  # the feasibility export functions read C.RESULTS
    missing = [str(p) for p in C.RESULTS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"layer {layer}: missing " + ", ".join(missing))
    results = {m: E.export_results(m) for m in C.MODELS}
    vbd = E.export_vbd_trials(results)
    allres = pd.concat(results.values(), ignore_index=True)

    # Figure 6 (notebook cell "reg = ..."): regular -> regular, the three common endings
    reg = vbd[(vbd.experiment == "regular") & (vbd.class_from == "regular") & ~vbd.authors_excluded].copy()
    reg3 = reg[reg.allomorph_from_canon.isin(ENDING_LABEL) & reg.allomorph_to_canon.isin(ENDING_LABEL)].copy()
    reg3["src"], reg3["tgt"] = reg3.allomorph_from_canon.map(ENDING_LABEL), reg3.allomorph_to_canon.map(ENDING_LABEL)
    grid = C.rank_summary(reg3, ["model", "src", "tgt"], n=n_boot)

    # Figure 7 (notebook cell "# Figure 7: ...")
    flagged = set(C.SUPPLETIVE_PAIRS) | set(C.HOMOGRAPH_PASTS) \
        | set(map(tuple, pairs.loc[pairs.inflected.isin(C.MODAL_PASTS) | (pairs.n_pasts_for_base > 1), ["base", "inflected"]].to_numpy()))
    vreg = vbd[(vbd.experiment == "regular") & ~vbd.authors_excluded]
    irr = vreg[vreg.class_from == "irregular"]
    is_flagged = np.array([(b, i) in flagged or (b2, i2) in flagged
                           for b, i, b2, i2 in zip(irr.base_from, irr.inflected_from, irr.base_to, irr.inflected_to)], dtype=bool)
    rres = allres[(allres.experiment == "regular") & ~allres.excluded]
    clean_regular = set(map(tuple, pairs.loc[pairs.auto_class.str.startswith("regular_"), ["base", "inflected"]].to_numpy()))
    vr = vreg[vreg.class_from == "regular"]
    is_clean = np.array([(b, i) in clean_regular and (b2, i2) in clean_regular
                         for b, i, b2, i2 in zip(vr.base_from, vr.inflected_from, vr.base_to, vr.inflected_to)], dtype=bool)
    conds = {
        COND_PAST_REG: vr,
        COND_PAST_REG_CLEAN: vr[is_clean],
        COND_PAST_IRR: irr,
        COND_PAST_IRR_CLEAN: irr[~is_flagged],
        COND_NNS: rres[(rres.group0 == "NNS") & (rres.group1 == True)],  # noqa: E712 (object column)
        COND_VBZ: rres[(rres.group0 == "VBZ") & (rres.group1 == True)],  # noqa: E712
        COND_RANDOM: allres[allres.experiment.isin(["random_to_NNS", "random_to_VBZ"])],
    }
    d7 = pd.concat([d.assign(condition=k) for k, d in conds.items()], ignore_index=True)
    cond = C.rank_summary(d7, ["model", "condition"], n=n_boot)

    # per-verb (notebook cell "# Worst / best target verbs ...")
    hom = pairs.drop_duplicates("inflected").set_index("inflected").past_homophones
    per_verb = reg3.groupby(["model", "tgt", "base_to", "inflected_to"]).gt_label_rank \
        .agg(mean_rank="mean", trials="size").reset_index()
    per_verb["past_sounds_like"] = per_verb.inflected_to.map(hom).fillna("")

    for df in (grid, cond, per_verb):
        df.insert(0, "layer", layer)
    return grid, cond, per_verb


# --------------------------------------------------------------------------- summaries

def best_layers(cond: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, condition), d in cond.groupby(["model", "condition"], sort=False):
        by_top1 = d.loc[d.top1.idxmax()]
        by_rank = d.loc[d.mean_rank.idxmin()]
        rows.append({"model": model, "condition": condition,
                     "best_layer_top1": int(by_top1.layer), "top1": by_top1.top1,
                     "top1_lo": by_top1.top1_lo, "top1_hi": by_top1.top1_hi,
                     "best_layer_mean_rank": int(by_rank.layer), "mean_rank": by_rank.mean_rank,
                     "mean_lo": by_rank.mean_lo, "mean_hi": by_rank.mean_hi})
    return pd.DataFrame(rows)


def layer8_vs_feasibility(grid: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame | None:
    """This run's layer 8 next to the feasibility demo's committed layer-8 tables (table7 = Figure 6, table9 = Figure 7)."""
    t7, t9 = C.TABLES / "table7_vbd_allomorph_grid.csv", C.TABLES / "table9_regular_vs_irregular.csv"
    if 8 not in set(grid.layer) or not (t7.exists() and t9.exists()):
        return None
    cols = ["mean_rank", "top1", "n_trials"]
    ours7 = grid[grid.layer == 8].assign(cell=lambda d: d.src + " → " + d.tgt)
    feas7 = pd.read_csv(t7).assign(cell=lambda d: d.src + " → " + d.tgt)
    ours9 = cond[cond.layer == 8].rename(columns={"condition": "cell"})
    feas9 = pd.read_csv(t9).rename(columns={"condition": "cell"})
    parts = []
    for fig, ours, feas in [("Figure 6", ours7, feas7), ("Figure 7", ours9, feas9)]:
        m = ours[["model", "cell", *cols]].merge(feas[["model", "cell", *cols]], on=["model", "cell"],
                                                 suffixes=("_ours", "_feasibility"), how="outer")
        parts.append(m.assign(figure=fig))
    out = pd.concat(parts, ignore_index=True)
    return out[["figure", "model", "cell", "top1_ours", "top1_feasibility", "mean_rank_ours", "mean_rank_feasibility",
                "n_trials_ours", "n_trials_feasibility"]]


# --------------------------------------------------------------------------- figures

# Okabe-Ito; validated (dataviz validate_palette.js); markers + end labels as secondary encoding
COND_STYLE = {
    COND_PAST_REG_CLEAN: ("#0072B2", "o", "-", "past, regular"),
    COND_PAST_IRR_CLEAN: ("#D55E00", "s", "-", "past, irregular"),
    COND_NNS: ("#009E73", "^", "-", "plural -s"),
    COND_VBZ: ("#CC79A7", "D", "-", "3sg -s"),
    COND_RANDOM: (C.CONTROL_COLOR, "x", "--", "random pair"),
}
ENDING_STYLE = {"[t]": ("#0072B2", "o"), "[d]": ("#E69F00", "s"), "[ɪd]": ("#009E73", "^")}


def _end_labels(ax, labels, min_gap_pt=11):
    """
    Direct labels at the right end of each line, pushed apart vertically so they don't overlap.
    `labels` = [(x_last, y_last, text)]; call after the axis limits and scale are final.
    """
    fig = ax.figure
    fig.canvas.draw()
    pts = [(ax.transData.transform((x, y))[1], x, y, t) for x, y, t in labels if np.isfinite(y)]
    if not pts:
        return
    pts.sort()
    gap = min_gap_pt * fig.dpi / 72
    ys = [p[0] for p in pts]
    for i in range(1, len(ys)):          # push up
        ys[i] = max(ys[i], ys[i - 1] + gap)
    lo = ax.transAxes.transform((0, 0))[1]
    shift = max(0.0, lo - ys[0])
    ys = [y + shift for y in ys]
    inv = ax.transData.inverted()
    for y_disp, (_, x, y, t) in zip(ys, pts):
        x_disp = ax.transData.transform((x, y))[0]
        y_lab = inv.transform((x_disp, y_disp))[1]
        ax.annotate(t, xy=(x, y), xytext=(x + 0.35, y_lab), textcoords="data", va="center", fontsize=8,
                    color="#333333", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": "#bbbbbb", "lw": 0.6, "shrinkA": 0, "shrinkB": 3})


def fig_conditions(cond: pd.DataFrame, layers: list[int]):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for col, model in enumerate(C.MODELS):
        for row, (est, lo, hi, ylabel) in enumerate([("top1", "top1_lo", "top1_hi", "top-1 accuracy (higher is better)"),
                                                     ("mean_rank", "mean_lo", "mean_hi", "mean rank (log; lower is better)")]):
            ax = axs[row, col]
            ends = []
            for condition, (color, marker, ls, label) in COND_STYLE.items():
                d = cond[(cond.model == model) & (cond.condition == condition)].set_index("layer").reindex(layers)
                x = np.array(layers)
                ax.fill_between(x, d[lo], d[hi], color=color, alpha=0.12, lw=0)
                ax.plot(x, d[est], ls, color=color, marker=marker, ms=6, lw=2, label=label)
                ends.append((x[-1], d[est].iloc[-1], label))
            if est == "mean_rank":
                ax.set_yscale("log")
            else:
                ax.set_ylim(0, 1.02)
            ax.set_ylabel(ylabel)
            ax.set_xticks(layers)
            if row == 0:
                ax.set_title(f"{model} ({'word probe' if model == 'Word' else 'raw hidden states'})")
            if row == 1:
                ax.set_xlabel("wav2vec2-base layer")
            ax.set_xlim(min(layers) - 0.3, max(layers) + 2.2)
            _end_labels(ax, ends)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Past tense vs. -s across layers (within-class analogies; bands = 95% item-bootstrap CI)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.text(0.5, -0.01, "past, regular = pairs whose automatic class is regular [t]/[d]/[ɪd]; "
             "past, irregular = without modals, be and homographs (as in the feasibility notebook, Figure 7)",
             ha="center", fontsize=8, color="#555555")
    save(fig, "fig_layers_conditions")


def fig_allomorph_top1(grid: pd.DataFrame, layers: list[int]):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    pending = []
    for r, model in enumerate(C.MODELS):
        for c, tgt in enumerate(ENDINGS):
            ax = axs[r, c]
            ends = []
            for src in ENDINGS:
                color, marker = ENDING_STYLE[src]
                d = grid[(grid.model == model) & (grid.src == src) & (grid.tgt == tgt)].set_index("layer").reindex(layers)
                x = np.array(layers)
                ax.fill_between(x, d.top1_lo, d.top1_hi, color=color, alpha=0.12, lw=0)
                ax.plot(x, d.top1, "-", color=color, marker=marker, ms=6, lw=2, label=f"source {src}")
                ends.append((x[-1], d.top1.iloc[-1], f"from {src}"))
            ax.set_ylim(0, 1.02)
            ax.set_xticks(layers)
            ax.set_xlim(min(layers) - 0.3, max(layers) + 2.4)
            pending.append((ax, ends))
            if r == 0:
                ax.set_title(f"target ending {tgt}")
            if c == 0:
                ax.set_ylabel(f"{model}\ntop-1 accuracy")
            if r == 1:
                ax.set_xlabel("wav2vec2-base layer")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Regular past tense: transfer across [t] / [d] / [ɪd], by layer (bands = 95% item-bootstrap CI)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ax, ends in pending:
        _end_labels(ax, ends)
    save(fig, "fig_layers_allomorph_top1")


def fig_allomorph_grids(grid: pd.DataFrame, layers: list[int]):
    """Figure 6 of the notebook for every layer: one 3x3 heatmap per layer, per model."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    ncol = 6
    nrow = int(np.ceil(len(layers) / ncol))
    norm = Normalize(vmin=0, vmax=1)
    for model in C.MODELS:
        fig, axs = plt.subplots(nrow, ncol, figsize=(2.5 * ncol + 1.2, 2.6 * nrow + 0.8), squeeze=False,
                                gridspec_kw={"wspace": 0.12, "hspace": 0.35})
        for k, ax in enumerate(axs.flat):
            if k >= len(layers):
                ax.axis("off")
                continue
            layer = layers[k]
            s = grid[(grid.model == model) & (grid.layer == layer)].set_index(["src", "tgt"])
            v = s.top1.unstack().reindex(index=ENDINGS, columns=ENDINGS)
            annot = v.map(lambda x: f"{x:.2f}" if np.isfinite(x) else "–")
            bottom, left = k >= len(layers) - ncol, k % ncol == 0
            im = C.heatmap(ax, v, annot, norm, cmap="Blues", title=f"layer {layer}", fontsize=8,
                           xlabel="target" if bottom else "", ylabel="source" if left else "")
            if not bottom:
                ax.set_xticklabels([])
            if not left:
                ax.set_yticklabels([])
        fig.colorbar(im, ax=axs, label="top-1 accuracy", shrink=0.8)
        fig.suptitle(f"{model}: regular past tense, top-1 accuracy per ending pair and layer")
        save(fig, f"fig_layers_allomorph_grids_{model.lower()}")


def save(fig, name: str):
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png/.pdf")


# --------------------------------------------------------------------------- main

def main(args):
    import matplotlib
    matplotlib.use("Agg")
    C.set_style()
    layers = parse_layers(args.layers)
    pairs = load_pairs()

    grids, conds, verbs = [], [], []
    for layer in layers:
        print(f"layer {layer} ...", flush=True)
        g, c, v = layer_tables(layer, pairs, args.n_boot)
        grids.append(g)
        conds.append(c)
        verbs.append(v)
    grid, cond, per_verb = (pd.concat(x, ignore_index=True) for x in (grids, conds, verbs))

    TABLES.mkdir(parents=True, exist_ok=True)
    grid.to_csv(TABLES / "layers_allomorph_grid.csv", index=False)
    cond.to_csv(TABLES / "layers_conditions.csv", index=False)
    per_verb.to_csv(TABLES / "layers_per_verb.csv.gz", index=False)
    best = best_layers(cond)
    best.to_csv(TABLES / "layers_best.csv", index=False)
    cmp8 = layer8_vs_feasibility(grid, cond)
    if cmp8 is not None:
        cmp8.to_csv(TABLES / "layer8_vs_feasibility.csv", index=False)
    print("  wrote tables/layers_*.csv" + (", tables/layer8_vs_feasibility.csv" if cmp8 is not None else ""))

    print("figures ...")
    fig_conditions(cond, layers)
    fig_allomorph_top1(grid, layers)
    fig_allomorph_grids(grid, layers)

    pd.set_option("display.width", 160)
    print("\nBest layer per condition (top-1 accuracy):")
    print(best[["model", "condition", "best_layer_top1", "top1", "best_layer_mean_rank", "mean_rank"]]
          .round(3).to_string(index=False))
    if cmp8 is not None:
        print("\nLayer 8: this run vs. the feasibility demo's layer-8 results (expect close, not identical):")
        print(cmp8.round(3).to_string(index=False))
    print(f"\nDone. Everything is in {OUT.relative_to(C.ROOT)}/")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__, formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(prog))
    p.add_argument("--layers", default="0-11", help="e.g. '0-11', '8', '6,8,10'")
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples per cell (as in the notebook)")
    main(p.parse_args())
