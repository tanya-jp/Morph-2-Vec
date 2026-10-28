# Running Morph-2-Vec on Alliance Canada

This folder runs the full pipeline (`scripts/run_pipeline.py`) on an Alliance Canada cluster with SLURM. The goal is all 12 layers of wav2vec2-base on LibriSpeech `train-clean-100`, the paper's Fig. 1 setup. Our CAS 784 project uses those runs to compare English past tense (VBD) across layers.

## Quick start: test that everything works

On a login node (Fir: `ssh <username>@fir.alliancecan.ca`):

```bash
git clone -b alliance https://github.com/tanya-jp/Morph-2-Vec.git ~/Morph-2-Vec
cd ~/Morph-2-Vec
echo 'export M2V_EMAIL="you@mcmaster.ca"' > alliance/local.sh   # optional: job emails
tmux new -s m2v                          # keeps the next two steps running if SSH drops
bash alliance/setup_env.sh               # once; ~20-30 min
bash alliance/prefetch.sh test-clean     # small test data, a few minutes
bash alliance/smoke.sh                   # submits the test job (~1 h on a GPU)
```

When the job has finished, `bash alliance/status.sh test-clean` should show `ok` in every column for layer 8. If it does, the full run is `bash alliance/prefetch.sh` followed by `bash alliance/submit_all.sh` (step 4 below).

The pipeline code is the same as on Bart's Windows machine. This folder adds:

| File | What it does | Where it runs |
|---|---|---|
| `config.sh` | Settings: account, paths, split, layers, GPU request | sourced by everything |
| `setup_env.sh` | Python environment, spaCy model, `data/` and `outputs/` linked to `/scratch` | login node, once |
| `prefetch.sh` | Downloads everything the jobs need, because compute nodes may have no internet | login node |
| `smoke.sh` | Checks setup and data are in place, then submits `jobs/smoke.sbatch` | login node |
| `jobs/smoke.sbatch` | Whole pipeline on `test-clean`, layer 8, tiny probe training | GPU job, ~1 h |
| `jobs/prepare.sbatch` | Extract audio + preprocessing notebook | CPU job |
| `jobs/extract.sbatch` | Hidden states for all layers in one pass + layer-independent stages | GPU job |
| `jobs/layer.sbatch` | Probe training, embeddings, analogy runs; one array task per layer | GPU jobs |
| `submit_all.sh` | Submits prepare → extract → layers as a dependency chain | login node |
| `status.sh` | Shows which stages are done, per layer, and disk usage | login node |

Code changes outside this folder:
- `scripts/run_pipeline.py` has a new `fetch` stage that does all the downloading: archives, spelling map, CMUdict and model weights.
- It also counts CPUs from the SLURM allocation rather than the whole node.
- `notebooks/preprocessing/librispeech.ipynb` reads the cached spelling map and CMUdict from `data/external/` when present, so it runs offline.

## Which cluster

Any cluster with GPUs works: Nibi, Fir, Rorqual, Narval, or Trillium's GPU partition. The scripts work whether or not compute nodes have internet. The account is `def-brodbeck`.

**GPU type.** Some clusters, Fir among them, refuse a GPU job that doesn't name the GPU type. `config.sh` picks one automatically from what the cluster offers:
1. A 40 GB slice of an H100 (MIG `3g.40gb`). This is plenty for wav2vec2-base (Bart used a 12 GB GPU), and slices usually start sooner.
2. Otherwise, a whole H100.
3. Otherwise, any GPU.

To choose yourself, set `M2V_GPUS` in `alliance/local.sh`, e.g. `export M2V_GPUS="h100:1"` for a whole H100. If `sbatch` rejects the memory or CPU request for a slice, a whole H100 is the fix. Submit through `smoke.sh` and `submit_all.sh`; they pass the GPU type. A plain `sbatch alliance/jobs/*.sbatch` asks for "any GPU", which Fir refuses.

**Login node trouble.** If a login node gives `Input/output error`s on files under `/cvmfs` (we hit this on Fir's `login2`), switch nodes with `ssh login1` and work there.

On Trillium, jobs must be submitted from `$SCRATCH`. Clone the repo there and set `M2V_REPO` accordingly.

## Storage

| What | Where | Size (train-clean-100) |
|---|---|---|
| Repo (code) | `$HOME/Morph-2-Vec` | small |
| Virtual environment | `$HOME/venvs/morph2vec` | ~6 GB |
| `data/` (audio, alignments, archives) | `$SCRATCH/morph2vec/data` via symlink | ~21 GB + test-clean |
| `outputs/` | `$SCRATCH/morph2vec/outputs` via symlink | ~40 GB for the shared stages + ~55 GB per layer (hidden states + embeddings) → **~700 GB for 12 layers** |
| HF cache (wav2vec2 weights) | `$SCRATCH/morph2vec/hf_cache` | <1 GB |

- Check your quotas with `diskusage_report` before the full run.
- `/scratch` deletes files that haven't been used for a while; you get an email first. **Copy results you want to keep** to `$HOME` or `/project`: `outputs/analogy/runs*`, `outputs/models`, the logs, and any figures.
- The hidden-state `.h5` files can be regenerated, so there's no need to back them up.

## Step by step

### 1. Clone and set up (login node, once)

```bash
ssh tanyajp@<cluster>.alliancecan.ca
git clone -b alliance https://github.com/tanya-jp/Morph-2-Vec.git ~/Morph-2-Vec
cd ~/Morph-2-Vec
bash alliance/setup_env.sh
```

This loads `StdEnv/2023 gcc python/3.11 arrow` and creates the virtualenv. It installs PyTorch from the Alliance wheelhouse (`pip install --no-index torch`), then `requirements.txt` and `en_core_web_trf`, and links `data/` and `outputs/` to scratch. It ends with an import check.

### Email notifications (optional)

Put your email in `alliance/local.sh`. That file is gitignored, so your address stays out of the repo:

```bash
echo 'export M2V_EMAIL="you@mcmaster.ca"' > alliance/local.sh
```

Every job then emails you when it starts, finishes, fails, or has used 80% of its time limit. The 12 layer jobs send one email per layer when it finishes or fails, without the "started" emails. Change what triggers an email with `M2V_MAIL_TYPE` in `config.sh`.

### 2. Download (login node)

```bash
tmux new -s fetch            # survives a dropped SSH connection; reattach: tmux attach -t fetch
bash alliance/prefetch.sh    # test-clean + train-clean-100, alignments, spelling map, CMUdict, wav2vec2-base
```

It ends by checking that wav2vec2-base and the spaCy model load with the network disabled.

### 3. Smoke test

```bash
bash alliance/smoke.sh       # prints the job number, the live-log command and how to check the result
squeue -u $USER              # PD = waiting, R = running; logs are in $SCRATCH/morph2vec/logs
bash alliance/status.sh test-clean
```

The smoke test trains its probe on only 1000 frames. Before a real `test-clean` run, delete `outputs/models/librispeech-test-clean` and the dependent `outputs/model_embeddings/librispeech-test-clean` and `outputs/analogy/runs/librispeech-test-clean`.

### 4. Full run

```bash
bash alliance/submit_all.sh
bash alliance/status.sh      # check progress any time; logs are in $SCRATCH/morph2vec/logs
```

| Job | Resources | Expected time (Bart's RTX 3080 Ti; an H100 should be faster) |
|---|---|---|
| prepare | 16 CPUs, 64 GB, no GPU | ~2 h |
| extract | 1 GPU, 8 CPUs, 64 GB | ~4 h |
| layer × 12 (6 at a time) | 1 GPU, 8 CPUs, 64 GB each | ~3.5 h each |

Each layer job copies its ~52 GB hidden-state file to the node-local disk (`$SLURM_TMPDIR`) for fast random reads. The disk is printed at the top of each log.

### If something fails

- The runner skips finished stages and redoes half-finished ones. Fix the problem and re-submit only what's left, for example:
  - `bash alliance/submit_all.sh --from layers`
  - `M2V_LAYERS=3,7 bash alliance/submit_all.sh --from layers`
- **Out of memory:** look for `oom-kill` or `OUT_OF_MEMORY` with `seff <jobid>`, then raise `--mem`, e.g. `sbatch --mem=96G ...`. The one-pass extraction of 12 layers is the most likely to need more.
- **Timeout:** raise `--time`, or re-submit. The analogy step resumes from `partial_results/`.
- **A network error inside a job:** something wasn't cached. Run `prefetch.sh` again and look at what it downloads.
- **`ModuleNotFoundError` for pyarrow, or a pyarrow version conflict:** run `module spider arrow` and load a version that lists Python 3.11.
- **Interactive debugging:** `salloc --time=1:00:00 --gpus-per-node=1 --cpus-per-task=4 --mem=16G`, then:
  ```bash
  source alliance/config.sh && m2v_activate
  export SLURM_TMPDIR=${SLURM_TMPDIR:-/tmp}
  m2v_job_env
  ```

## After the run

```bash
for L in $(seq 0 11); do
    python scripts/summarize_analogy_results.py --dataset librispeech-train-clean-100 --layer $L
done
```

The past-tense feasibility scripts (`scripts/feasibility/`) are still fixed to layer 8 (`BASE_MODEL` in `common.py`). Extending them to all layers is the next step.
