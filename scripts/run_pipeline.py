"""
Cross-platform runner for the Morph-2-Vec analogy pipeline.

Replaces the bash-based Snakemake rules (Snakefile, workflows/librispeech/Snakefile)
so the pipeline runs natively on Windows and Linux (incl. Alliance Canada clusters;
see alliance/README.md). It calls the same scripts / notebooks with
the same Hydra overrides and writes to the same output paths, so the summary
notebooks (`summary_notebooks/analogy v4*.ipynb`) can read the results unchanged.

Stages (run in this order by `--stage all`):

    fetch       everything that needs the internet: LibriSpeech + alignment archives,
                the British->American spelling map, CMUdict and the base model weights
                (run it alone on a login node when compute nodes are offline)
    download    extract the archives; FLAC -> 16 kHz WAV
    preprocess  notebooks/preprocessing/librispeech.ipynb   -> outputs/preprocessed_data
    extract     scripts/extract_hidden_states.py (all layers in one pass) -> outputs/hidden_states
    specs       scripts/generate_state_space_specs.py       -> outputs/state_space_specs
    equiv       scripts/make_equivalence_dataset.py          -> outputs/equivalence_datasets
    pos         scripts/make_pos_counts.py                   -> outputs/pos_counts
    inputs      notebooks/analogy/prepare_inputs.ipynb      -> outputs/analogy/inputs
    train       train_decoder.py (word probe, per layer)     -> outputs/models
    embed       scripts/extract_model_embeddings.py          -> outputs/model_embeddings
    analogy     notebooks/analogy/run.ipynb for the probe (runs/) and raw model (runs_id/)

Every stage is skipped when its outputs are already complete (a `.done` marker is
written next to each output), so the runner can be re-run after a failure.

Examples:

    # smoke test: everything on test-clean, layer 8, tiny probe training run
    python scripts/run_pipeline.py --split test-clean --layers 8 --smoke

    # main reproduction (paper Figs 2-7): layer 8 on train-clean-100
    python scripts/run_pipeline.py --split train-clean-100 --layers 8 --hot-dir F:/morph2vec_hot

    # Fig. 1: all layers
    python scripts/run_pipeline.py --split train-clean-100 --layers 0-11 --hot-dir F:/morph2vec_hot --hot-cleanup

    # held-out evaluation (App. B.2): probe trained on train-clean-100, analyzed on test-clean
    python scripts/run_pipeline.py --split test-clean --train-split train-clean-100 --layers 8
"""

from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Callable, Optional
import zipfile

REPO_ROOT = Path(__file__).resolve().parent.parent

LIBRISPEECH_URL_FORMAT = "https://www.openslr.org/resources/12/{split}.tar.gz"
LIBRISPEECH_ALIGNMENTS_URL = "https://zenodo.org/records/2619474/files/librispeech_alignments.zip?download=1"
# small text resources read by notebooks/preprocessing/librispeech.ipynb; cached here so
# preprocessing can run on compute nodes without internet access
EXTERNAL_RESOURCES = {
    "data/external/british_spellings.json":
        "https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/british_spellings.json",
    "data/external/cmudict.dict":
        "https://github.com/cmusphinx/cmudict/raw/master/cmudict.dict",
}

EQUIVALENCE = "word_broad_10frames_fixedlen25"
STAGES = ["fetch", "download", "preprocess", "extract", "specs", "equiv", "pos", "inputs",
          "train", "embed", "analogy"]


# --------------------------------------------------------------------------- utils

def log(msg: str):
    print(f"[run_pipeline] {msg}", flush=True)


def hydra_str(path) -> str:
    """Quote a path for use as a Hydra override value."""
    return "'" + str(path).replace("\\", "/") + "'"


def is_done(output: Path) -> bool:
    return output.exists() and Path(str(output) + ".done").exists()


def mark_done(output: Path):
    Path(str(output) + ".done").touch()


def run_stage(name: str, output: Path, fn: Callable[[], None], force: bool = False):
    """Run `fn` unless `output` is complete. Stale partial outputs are removed first."""
    if is_done(output) and not force:
        log(f"{name}: up to date ({output})")
        return
    if output.exists():
        log(f"{name}: removing incomplete output {output}")
        shutil.rmtree(output) if output.is_dir() else output.unlink()
    log(f"{name}: running -> {output}")
    fn()
    if not output.exists():
        raise RuntimeError(f"{name} finished but did not produce {output}")
    mark_done(output)


def run_python(args: list[str]):
    cmd = [sys.executable, *args]
    log(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_notebook(notebook: str, output_notebook: Path, parameters: dict):
    import papermill
    output_notebook.parent.mkdir(parents=True, exist_ok=True)
    log(f"papermill {notebook} -> {output_notebook} {parameters}")
    try:
        papermill.execute_notebook(
            str(REPO_ROOT / notebook), str(output_notebook),
            parameters=parameters, cwd=str(REPO_ROOT),
            log_output=True, autosave_cell_every=30)
    finally:
        clean_datasets_tmp()


def set_tmp_dir(tmp_dir: Path):
    """
    Route all temporary files (incl. `datasets` map results when caching is disabled,
    which can be tens of GB) to `tmp_dir`. TMPDIR takes precedence over TEMP/TMP in
    Python's `tempfile`, so all three are set.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for var in ("TMPDIR", "TEMP", "TMP"):
        os.environ[var] = str(tmp_dir)


def clean_datasets_tmp():
    """
    `datasets.disable_caching()` writes map results to `hf_datasets-*` temp dirs, which
    it fails to delete on Windows (the arrow files are still memory-mapped). Remove them
    once the notebook kernel holding them has exited.
    """
    for d in Path(os.environ["TMPDIR"]).glob("hf_datasets-*"):
        shutil.rmtree(d, ignore_errors=True)


def parse_layers(spec: str) -> list[int]:
    layers = []
    for part in spec.split(","):
        if "-" in part:
            start, end = part.split("-")
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


# --------------------------------------------------------------------------- paths

class Paths:
    def __init__(self, base_model_class: str, hot_dir: Optional[Path]):
        self.base_model_class = base_model_class
        self.hot_dir = hot_dir

    def base_model(self, layer: int) -> str:
        return f"{self.base_model_class}_{layer}"

    @staticmethod
    def dataset(split: str) -> str:
        return f"librispeech-{split}"

    def raw_split(self, split: str) -> Path:
        return Path("data/librispeech") / split

    def alignments(self) -> Path:
        return Path("data/librispeech_alignments")

    def preprocessed(self, split: str) -> Path:
        return Path("outputs/preprocessed_data") / self.dataset(split)

    def hidden_states(self, split: str, layer) -> Path:
        return Path("outputs/hidden_states") / f"{self.base_model_class}_{layer}" / f"{self.dataset(split)}.h5"

    def state_space_specs(self, split: str, layer: int) -> Path:
        return Path("outputs/state_space_specs") / self.dataset(split) / self.base_model(layer) / "state_space_specs.h5"

    def equivalence(self, split: str) -> Path:
        # Equivalence classing only depends on frame alignment, which is identical across
        # layers, so it is computed once per dataset and shared by all layers.
        return Path("outputs/equivalence_datasets") / self.dataset(split) / self.base_model_class / EQUIVALENCE / "equivalence.pkl"

    def pos_counts(self, split: str) -> Path:
        return Path("outputs/pos_counts") / f"{self.dataset(split)}.pkl"

    def analogy_inputs(self, split: str) -> Path:
        return Path("outputs/analogy/inputs") / self.dataset(split) / self.base_model_class

    def model_dir(self, train_split: str, layer: int, model: str) -> Path:
        return Path("outputs/models") / self.dataset(train_split) / self.base_model(layer) / model / EQUIVALENCE

    def embeddings(self, train_split: str, split: str, layer: int, model: str) -> Path:
        return Path("outputs/model_embeddings") / self.dataset(train_split) / self.base_model(layer) / model / EQUIVALENCE / f"{self.dataset(split)}.npy"

    def analogy_run(self, train_split: str, split: str, layer: int, model: str) -> Path:
        return Path("outputs/analogy/runs") / self.dataset(train_split) / self.base_model(layer) / model / EQUIVALENCE / self.dataset(split)

    def analogy_run_id(self, split: str, layer: int) -> Path:
        return Path("outputs/analogy/runs_id") / self.dataset(split) / self.base_model(layer)

    def hot(self, path: Path) -> Path:
        """
        Return a copy of `path` on the fast `hot_dir` disk (copying if needed), for
        stages that do random-access reads of large HDF5 files.
        """
        if self.hot_dir is None:
            return path
        hot_path = self.hot_dir / path
        if not hot_path.exists() or hot_path.stat().st_size != (REPO_ROOT / path).stat().st_size:
            log(f"copying {path} -> {hot_path}")
            hot_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / path, hot_path)
        return hot_path

    def drop_hot(self, path: Path):
        if self.hot_dir is not None and (self.hot_dir / path).exists():
            log(f"removing hot copy {self.hot_dir / path}")
            (self.hot_dir / path).unlink()


# --------------------------------------------------------------------------- download

def download(url: str, dest: Path, max_retries: int = 5):
    """Download `url` to `dest`, resuming partial downloads (Zenodo drops long connections)."""
    if dest.exists():
        return
    import requests
    from tqdm.auto import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"downloading {url}")

    for attempt in range(max_retries):
        offset = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0  # server ignored the range request; start over
                total = int(response.headers.get("Content-Length", 0)) + offset or None
                with open(tmp, "ab" if offset else "wb") as f, \
                        tqdm(total=total, initial=offset, unit="B", unit_scale=True, desc=dest.name) as bar:
                    for chunk in response.iter_content(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp.rename(dest)
            return
        except requests.RequestException as e:
            log(f"download interrupted ({e}); retrying ({attempt + 1}/{max_retries})")
    raise RuntimeError(f"failed to download {url}")


def stage_fetch(paths: Paths, split: str, layer: int):
    """Download everything that needs the network, without extracting or processing it."""
    downloads = Path("data/downloads")
    if not is_done(paths.raw_split(split)):
        download(LIBRISPEECH_URL_FORMAT.format(split=split), downloads / f"{split}.tar.gz")
    if not is_done(paths.alignments()):
        download(LIBRISPEECH_ALIGNMENTS_URL, downloads / "librispeech_alignments.zip")
    for dest, url in EXTERNAL_RESOURCES.items():
        download(url, Path(dest))

    # base model weights -> Hugging Face cache ($HF_HOME)
    from omegaconf import OmegaConf
    model_ref = OmegaConf.load(REPO_ROOT / "conf/base_model" / f"{paths.base_model(layer)}.yaml").model_ref
    log(f"caching base model weights {model_ref}")
    import transformers
    transformers.AutoModel.from_pretrained(model_ref)  # downloads exactly the files it loads


def available_cpus() -> int:
    """CPUs this process may use (the SLURM allocation on a cluster, not the whole node)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # Windows / macOS
        return os.cpu_count() or 2


def flac_to_wav(flac_path: str):
    import soundfile as sf
    audio, sample_rate = sf.read(flac_path, dtype="int16")
    assert sample_rate == 16000, f"unexpected sample rate {sample_rate} in {flac_path}"
    sf.write(str(Path(flac_path).with_suffix(".wav")), audio, sample_rate, subtype="PCM_16")
    os.remove(flac_path)


def stage_download(paths: Paths, split: str, num_workers: int):
    downloads = Path("data/downloads")

    def download_split():
        archive = downloads / f"{split}.tar.gz"
        download(LIBRISPEECH_URL_FORMAT.format(split=split), archive)
        out = paths.raw_split(split)
        tmp = out.with_name(out.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        log(f"extracting {archive}")
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")

        flacs = [str(p) for p in tmp.glob("**/*.flac")]
        log(f"converting {len(flacs)} FLAC files to WAV")
        from tqdm.auto import tqdm
        with Pool(num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(flac_to_wav, flacs, chunksize=32), total=len(flacs)):
                pass
        tmp.rename(out)

    run_stage(f"download {split}", paths.raw_split(split), download_split)

    def download_alignments():
        archive = downloads / "librispeech_alignments.zip"
        download(LIBRISPEECH_ALIGNMENTS_URL, archive)
        log(f"extracting {archive}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(paths.alignments())

    run_stage("download alignments", paths.alignments(), download_alignments)


def find_alignment_dir(paths: Paths, split: str) -> Path:
    """The directory containing `{split}/{speaker}/{chapter}/*.TextGrid`."""
    root = paths.alignments()
    for candidate in [root, *sorted(p for p in root.glob("**/") if p.is_dir())]:
        if (candidate / split).is_dir():
            return candidate
    raise FileNotFoundError(f"could not find alignments for {split} under {root}")


# --------------------------------------------------------------------------- stages

def stage_preprocess(paths: Paths, split: str):
    out = paths.preprocessed(split)
    run_stage(f"preprocess {split}", out, lambda: run_notebook(
        "notebooks/preprocessing/librispeech.ipynb",
        Path("outputs/preprocessing") / paths.dataset(split) / "notebook.ipynb",
        {"split": split,
         "data_dir": str(paths.raw_split(split)),
         "alignment_dir": str(find_alignment_dir(paths, split)),
         "out_path": str(out)}))


def stage_extract(paths: Paths, split: str, layers: list[int], device: str):
    missing = [layer for layer in layers if not is_done(paths.hidden_states(split, layer))]
    if not missing:
        log(f"extract {split}: up to date for layers {layers}")
        return

    for layer in missing:
        out = paths.hidden_states(split, layer)
        if out.exists():
            out.unlink()

    outdir = Path("outputs/hidden_states") / f"_logs_{paths.dataset(split)}"
    log(f"extract {split}: layers {missing} in one pass")
    run_python([
        "scripts/extract_hidden_states.py",
        f"hydra.run.dir={hydra_str(outdir)}",
        f"base_model={paths.base_model(missing[0])}",
        f"+base_model.extract_layers=[{','.join(map(str, missing))}]",
        f"+base_model.hidden_state_path={hydra_str(paths.hidden_states(split, '{layer}'))}",
        f"dataset.processed_data_dir={hydra_str(paths.preprocessed(split))}",
        f"device={device}",
    ])
    for layer in missing:
        mark_done(paths.hidden_states(split, layer))


def stage_specs(paths: Paths, split: str, layer: int, parallel: bool):
    out = paths.state_space_specs(split, layer)
    run_stage(f"specs {split}", out, lambda: run_python([
        "scripts/generate_state_space_specs.py",
        f"hydra.run.dir={hydra_str(out.parent)}",
        f"base_model={paths.base_model(layer)}",
        f"dataset.processed_data_dir={hydra_str(paths.preprocessed(split))}",
        f"+base_model.hidden_state_path={hydra_str(paths.hidden_states(split, layer))}",
        f"+analysis.state_space_specs_path={hydra_str(out)}",
        f"+analysis.scheduler={'distributed' if parallel else 'synchronous'}",
    ]))


def stage_equiv(paths: Paths, split: str, layer: int):
    out = paths.equivalence(split)
    run_stage(f"equivalence {split}", out, lambda: run_python([
        "scripts/make_equivalence_dataset.py",
        f"hydra.run.dir={hydra_str(out.parent)}",
        f"base_model={paths.base_model(layer)}",
        f"+base_model.hidden_state_path={hydra_str(paths.hidden_states(split, layer))}",
        f"equivalence={EQUIVALENCE}",
        f"dataset.processed_data_dir={hydra_str(paths.preprocessed(split))}",
    ]))


def stage_pos(paths: Paths, split: str, pos_model: str):
    out = paths.pos_counts(split)
    run_stage(f"pos counts {split}", out, lambda: run_python([
        "scripts/make_pos_counts.py", str(paths.preprocessed(split)),
        "--output", str(out), "--model", pos_model,
    ]))


def stage_inputs(paths: Paths, split: str, spec_layer: int):
    outdir = paths.analogy_inputs(split)
    run_stage(f"analogy inputs {split}", outdir, lambda: run_notebook(
        "notebooks/analogy/prepare_inputs.ipynb", outdir / "prepare_inputs.ipynb",
        {"output_dir": str(outdir),
         "state_space_specs_path": str(paths.state_space_specs(split, spec_layer)),
         "pos_counts_path": str(paths.pos_counts(split))}))


def stage_train(paths: Paths, train_split: str, layer: int, model: str, device: str, smoke: bool):
    out = paths.model_dir(train_split, layer, model)

    def train():
        hidden_states = paths.hot(paths.hidden_states(train_split, layer))
        args = [
            "train_decoder.py",
            f"hydra.run.dir={hydra_str(out)}",
            f"dataset.processed_data_dir={hydra_str(paths.preprocessed(train_split))}",
            f"base_model={paths.base_model(layer)}",
            f"+base_model.hidden_state_path={hydra_str(hidden_states)}",
            f"model={model}",
            f"equivalence={EQUIVALENCE}",
            f"+equivalence.path={hydra_str(paths.equivalence(train_split))}",
            f"device={device}",
        ]
        if smoke:
            # restrict training to the first 1000 frames (see src/train.py)
            args.append("+smoke_test=true")
        run_python(args)

    run_stage(f"train {paths.base_model(layer)} {model} on {train_split}", out, train)


def stage_embed(paths: Paths, train_split: str, split: str, layer: int, model: str, device: str):
    out = paths.embeddings(train_split, split, layer, model)
    run_stage(f"embed {paths.base_model(layer)} {model} on {split}", out, lambda: run_python([
        "scripts/extract_model_embeddings.py",
        f"hydra.run.dir={hydra_str(out.parent)}",
        f"model={model}",
        f"+model.output_dir={hydra_str(paths.model_dir(train_split, layer, model))}",
        f"+model.embeddings_path={hydra_str(out)}",
        f"base_model={paths.base_model(layer)}",
        f"+base_model.hidden_state_path={hydra_str(paths.hot(paths.hidden_states(split, layer)))}",
        f"equivalence={EQUIVALENCE}",
        f"+equivalence.path={hydra_str(paths.equivalence(split))}",
        f"device={device}",
    ]))


def analogy_parameters(paths: Paths, split: str, outdir: Path, hidden_states: Path, embeddings: str) -> dict:
    inputs = paths.analogy_inputs(split)
    return {
        "output_dir": str(outdir),
        "hidden_states_path": str(hidden_states),
        "state_space_specs_path": str(inputs / "state_space_spec.h5"),
        "embeddings_path": embeddings,
        "inflection_results_path": str(inputs / "inflection_results.parquet"),
        "all_cross_instances_path": str(inputs / "all_cross_instances.parquet"),
        "most_common_allomorphs_path": str(inputs / "most_common_allomorphs.csv"),
        "false_friends_path": str(inputs / "false_friends.csv"),
    }


def stage_analogy(paths: Paths, train_split: str, split: str, layer: int, model: str):
    # word probe
    outdir = paths.analogy_run(train_split, split, layer, model)
    run_stage(f"analogy {paths.base_model(layer)} {model} on {split}", outdir / "experiment_results.csv",
              lambda: run_notebook(
                  "notebooks/analogy/run.ipynb", outdir / "run.ipynb",
                  analogy_parameters(paths, split, outdir, paths.hidden_states(split, layer),
                                     str(paths.embeddings(train_split, split, layer, model)))))

    # raw base-model hidden states ("identity" model)
    outdir = paths.analogy_run_id(split, layer)
    run_stage(f"analogy {paths.base_model(layer)} identity on {split}", outdir / "experiment_results.csv",
              lambda: run_notebook(
                  "notebooks/analogy/run.ipynb", outdir / "run.ipynb",
                  analogy_parameters(paths, split, outdir, paths.hot(paths.hidden_states(split, layer)), "ID")))


# --------------------------------------------------------------------------- main

def main(args):
    os.chdir(REPO_ROOT)
    # make `src` importable from scripts and notebook kernels
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    set_tmp_dir(Path(args.tmp_dir).resolve())

    layers = parse_layers(args.layers)
    split = args.split
    train_split = args.train_split or split
    splits = list(dict.fromkeys([train_split, split]))
    # word spans / equivalence classes do not depend on the layer
    spec_layer = 8 if 8 in layers else layers[0]

    stages = STAGES if args.stage == "all" else args.stage.split(",")
    for stage in stages:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage}; choose from {STAGES}")
    paths = Paths(args.base_model_class, Path(args.hot_dir) if args.hot_dir else None)

    for s in splits:
        if "fetch" in stages:
            stage_fetch(paths, s, spec_layer)
        if "download" in stages:
            stage_download(paths, s, args.num_workers)
        if "preprocess" in stages:
            stage_preprocess(paths, s)
        if "extract" in stages:
            stage_extract(paths, s, sorted(set(layers) | {spec_layer}), args.device)
        if "specs" in stages:
            stage_specs(paths, s, spec_layer, args.parallel_specs)
        if "equiv" in stages:
            stage_equiv(paths, s, spec_layer)

    if "pos" in stages:
        stage_pos(paths, split, args.pos_model)
    if "inputs" in stages:
        stage_inputs(paths, split, spec_layer)

    for layer in layers:
        if "train" in stages:
            stage_train(paths, train_split, layer, args.model, args.device, args.smoke)
        if "embed" in stages:
            stage_embed(paths, train_split, split, layer, args.model, args.device)
        if "analogy" in stages:
            stage_analogy(paths, train_split, split, layer, args.model)
        if args.hot_cleanup:
            for s in splits:
                paths.drop_hot(paths.hidden_states(s, layer))


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__, formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(prog))
    p.add_argument("--split", default="test-clean",
                   help="LibriSpeech split to analyze (e.g. test-clean, train-clean-100)")
    p.add_argument("--train-split", default=None,
                   help="split on which to train the word probe (default: --split)")
    p.add_argument("--layers", default="8", help="layers, e.g. '8', '0-11', '6,8,10'")
    p.add_argument("--stage", default="all", help=f"'all' or comma-separated subset of {STAGES}")
    p.add_argument("--model", default="ffff_32", help="probe config in conf/model (paper: ffff_32)")
    p.add_argument("--base-model-class", default="w2v2",
                   help="prefix of conf/base_model/{class}_{layer}.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true",
                   help="tiny probe training run, to check that the pipeline works end to end")
    p.add_argument("--hot-dir", default=None,
                   help="fast (SSD) directory for copies of HDF5 files read randomly by train/analogy")
    p.add_argument("--hot-cleanup", action="store_true",
                   help="delete each layer's hot copies once that layer is finished")
    p.add_argument("--parallel-specs", action="store_true",
                   help="compute state space specs with a multi-process dask cluster")
    p.add_argument("--pos-model", default="en_core_web_trf")
    p.add_argument("--tmp-dir", default=str(REPO_ROOT / "outputs" / "tmp"),
                   help="temporary files (can reach tens of GB during preprocessing)")
    p.add_argument("--num-workers", type=int, default=max(1, available_cpus() - 2))
    main(p.parse_args())
