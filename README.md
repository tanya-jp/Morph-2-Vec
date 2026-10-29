# Morph-2-Vec (CAS 784 fork)

This is a fork of the code for Gauthier, Breiss, Leonard & Chang (2025), *Emergent morpho-phonological representations in self-supervised speech models* (EMNLP 2025; PDF: `Project-Research-Paper_Morpho.pdf`). The original commits are by Jon Gauthier.

It is used for a CAS 784 course project (Group 3). The fork adds three things:
- **Windows support and memory fixes**, so the full pipeline runs on one desktop PC (32 GB RAM, 12 GB GPU);
- **a reproduction of the paper's main results** for wav2vec2-base layer 8 on LibriSpeech `train-clean-100`;
- **a past-tense pilot**: does the model represent English past tense ([t] / [d] / [ɪd], and irregular verbs) the way the paper found for -s?

Everything that was changed, run and found is documented in [`CAS784/Project_Feasibility/Project_Feasibility.md`](CAS784/Project_Feasibility/Project_Feasibility.md).

## Setup

You need a conda environment `morph2vec` (Python 3.11) built from `environment.yml` + `requirements.txt`, with a CUDA build of PyTorch. The exact steps and the environment variables to set are in [Project_Feasibility.md §5](CAS784/Project_Feasibility/Project_Feasibility.md#5-setup-environment-vs-code-machine). Run everything from the repo root.

## Running the pipeline

One command downloads LibriSpeech and the alignments, extracts the hidden states, trains the word probe, and runs the analogy experiments. It resumes after a failure:

```powershell
python scripts/run_pipeline.py --split train-clean-100 --layers 8 --hot-dir F:/morph2vec_hot
```

About 8–9 h and about 115 GB of disk for one layer; options are in §6 of Project_Feasibility.md.

**On Alliance Canada clusters** (Linux, SLURM), use the scripts in [`alliance/`](alliance/README.md): environment setup, offline prefetching, a smoke test, and a job chain for all 12 layers. Quick result tables:

```powershell
python scripts/summarize_analogy_results.py --dataset librispeech-train-clean-100 --layer 8
```

## Feasibility notebook (Milestone 3)

[`notebooks/feasibility/feasibility_demo.ipynb`](notebooks/feasibility/feasibility_demo.ipynb) is saved with its outputs, so it can be read on GitHub as is. It covers:
- the reproduction of the paper's Figs. 3–4;
- the baselines;
- the past-tense pilot;
- what we learned.

It reads only the small files in `outputs/feasibility/data/` and runs in under a minute without a GPU. To regenerate everything from the pipeline outputs:

```powershell
python scripts/feasibility/01_inventory_and_checks.py      # sanity checks  -> outputs/feasibility/checks/
python scripts/feasibility/02_export_compact_data.py       # compact data   -> outputs/feasibility/data/
python scripts/feasibility/03_prepare_theory_materials.py  # annotation sheets -> outputs/feasibility/theory/
python scripts/feasibility/build_notebook.py --execute     # rebuild + run the notebook
```

`scripts/feasibility/build_notebook.py` is the source of the notebook, so edit cells there rather than in the `.ipynb`.

**Running from the share bundle** (no pipeline outputs needed):
1. Unpack the bundle; it mirrors the repo layout.
2. Set `MORPH2VEC_ROOT` to the unpacked folder.
3. Run the last command above, or open the notebook in Jupyter.

`outputs/` and `data/` are gitignored. Figures and tables are written to `outputs/feasibility/figures/` and `tables/`.

## Past tense across layers

`scripts/layers/compare_layers.py` runs the past-tense analysis of the feasibility notebook (Part 4) on every wav2vec2-base layer. It uses the notebook's own code: Figure 6's [t] / [d] / [ɪd] grid, and Figure 7's regular vs. irregular past against plural -s, 3sg -s and the random-pair control.

It needs the analogy results of the layers: `outputs/analogy/runs*/…/w2v2_<layer>/…/experiment_results.csv`, plus `outputs/analogy/inputs/`. The full 12-layer run on Alliance produces them (`alliance/README.md`). To work on a laptop, copy them over:

```bash
rsync -avh --exclude 'partial_results/' <user>@fir.alliancecan.ca:/scratch/<user>/morph2vec/outputs/analogy/ outputs/analogy/
```

Then open **`notebooks/layers/past_tense_layers.ipynb`** and run it top to bottom; it takes about a minute, CPU only. Each step has a description: the data checks, a sanity check of our layer 8 against the feasibility run, the main figure, the best layer, the [t]/[d]/[ɪd] transfer, and the verbs that fail. The same analysis also runs without Jupyter: `python scripts/layers/compare_layers.py`.

It needs only pandas, numpy, matplotlib and seaborn. The output in `outputs/feasibility/layers/` is small and meant to be committed:
- `tables/layers_conditions.csv`, `tables/layers_allomorph_grid.csv`: mean rank and top-1 accuracy, with 95% item-bootstrap CIs, per layer and model.
- `tables/layers_best.csv`: the best layer per condition.
- `tables/layers_per_verb.csv.gz`: per-verb ranks.
- `tables/layer8_vs_feasibility.csv`: this run's layer 8 next to the feasibility demo's layer-8 results.
- `figures/`: the plots.

## Repository map

- **`scripts/`**: pipeline runner, summary script, and `feasibility/` (the notebook's helpers and export scripts).
- **`src/`**: probe model, datasets, state-space and analogy code.
- **`notebooks/`**:
  - `preprocessing/`, `analogy/`: pipeline stages run by the runner;
  - `feasibility/`: the Milestone 3 notebook.
- **`summary_notebooks/`**: the authors' figure notebooks. They are not adapted to our setup; they need 12 layers and WorldLex frequencies.
- **`conf/`**: Hydra configs (`model/ffff_32.yaml` is the paper's probe).
- **`CONTRIBUTIONS.md`**: the group's contribution log.
