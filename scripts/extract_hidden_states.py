from pathlib import Path
from typing import Optional

import datasets
import h5py
import numpy as np
import pyarrow as pa
import torch
import transformers

from beartype import beartype
import hydra
from omegaconf import DictConfig

from src.models.transformer import prepare_feature_extractor


class HiddenStateWriter:
    """
    Stream frame hidden states for one or more layers into HDF5 files, using the
    same schema as `SpeechHiddenStateDataset.to_hdf5`:

        states: num_frames * 1 * hidden_size
        compression_ratios: num_items
        flat_idxs: num_frames * 2 (item index, frame index)
        attrs["model_name"]

    Frames are written as they are produced, so memory use does not grow with
    the size of the corpus.
    """

    # Chunks of 256 frames (~768 KB at float32 / 768 dims) fit in h5py's default
    # chunk cache and keep random-access reads of short spans cheap.
    CHUNK_FRAMES = 256

    def __init__(self, paths: dict[int, str], model_name: str, hidden_size: int):
        self.paths = paths
        self.files = {}
        self.states = {}
        for layer, path in paths.items():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            f = h5py.File(path, "w")
            f.attrs["model_name"] = model_name
            self.files[layer] = f
            self.states[layer] = f.create_dataset(
                "states", shape=(0, 1, hidden_size), maxshape=(None, 1, hidden_size),
                dtype=np.float32, chunks=(self.CHUNK_FRAMES, 1, hidden_size))

        self.num_frames = 0
        # compact (item index, frame index) arrays per batch; a flat list of Python
        # tuples would take ~2 GB on train-clean-100
        self.flat_idxs: list[np.ndarray] = []
        self.compression_ratios: dict[int, float] = {}

    def write(self, layer_states: dict[int, torch.Tensor],
              flat_idxs: list[tuple[int, int]],
              compression_ratios: dict[int, float]):
        """
        Append a batch of frames. `layer_states` maps layer -> num_new_frames * hidden_size.
        """
        num_new = len(flat_idxs)
        for layer, states in layer_states.items():
            assert states.shape[0] == num_new
            dset = self.states[layer]
            dset.resize(self.num_frames + num_new, axis=0)
            dset[self.num_frames:] = states.unsqueeze(1).float().numpy()

        self.num_frames += num_new
        self.flat_idxs.append(np.array(flat_idxs, dtype=np.int32).reshape(-1, 2))
        self.compression_ratios.update(compression_ratios)

    def close(self, complete: bool = True):
        """
        Write item-level metadata and close all files. With `complete=False`
        (e.g. after an error), files are closed without metadata.
        """
        if complete:
            # need contiguous keys in compression_ratios
            assert set(self.compression_ratios.keys()) == set(range(len(self.compression_ratios)))
            compression_ratios = np.array([self.compression_ratios[idx] for idx in range(len(self.compression_ratios))])
            flat_idxs = np.concatenate(self.flat_idxs) if self.flat_idxs else np.zeros((0, 2), dtype=np.int32)
            assert len(flat_idxs) == self.num_frames

        for f in self.files.values():
            if complete:
                f.create_dataset("compression_ratios", data=compression_ratios, dtype=np.float32)
                f.create_dataset("flat_idxs", data=flat_idxs, dtype=np.int32)
            f.close()


@beartype
def extract_hidden_states(dataset: datasets.Dataset,
                          model: transformers.PreTrainedModel,
                          feature_extractor: transformers.SequenceFeatureExtractor,
                          writer: HiddenStateWriter,
                          layers: list[int],
                          pseudo_causal: bool = False,
                          batch_size=12) -> None:
    model.eval()

    def collate_batch(batch):
        batch = feature_extractor.pad(
            [{"input_values": values_i} for values_i in batch["input_values"]],
            max_length=None,
            return_tensors="pt",
            return_attention_mask=True)
        return batch

    def extract_representations(batch_items, idxs):
        batch = collate_batch(batch_items)

        with torch.no_grad():
            output = model(output_hidden_states=True,
                           input_values=batch["input_values"].to(model.device),
                           attention_mask=batch["attention_mask"].to(model.device))

        input_lengths = batch["attention_mask"].sum(dim=1)
        frame_lengths = model._get_feat_extract_output_lengths(input_lengths)

        batch_compression_ratios = (frame_lengths / input_lengths.numpy())

        flat_idxs = []
        compression_ratios = {}
        layer_states = {layer: [] for layer in layers}
        for b, (idx, num_frames_i, compression_i) in enumerate(zip(idxs, frame_lengths, batch_compression_ratios)):
            flat_idxs.extend([(idx, j) for j in range(num_frames_i)])
            compression_ratios[idx] = compression_i.item()
            for layer in layers:
                # batch_size * sequence_length * hidden_size
                layer_states[layer].append(output.hidden_states[layer][b, :num_frames_i].cpu())

        writer.write({layer: torch.cat(states, dim=0) for layer, states in layer_states.items()},
                     flat_idxs, compression_ratios)

        # On Windows (WDDM), GPU memory held by PyTorch's caching allocator is charged
        # against system commit; release it so it does not pin ~10 GB between batches.
        del output
        if model.device.type == "cuda":
            torch.cuda.empty_cache()

    def extract_representations_pseudo_causal(item, idx, max_length=None, frame_counts=None):
        assert frame_counts is not None
        assert max_length is not None

        audio = torch.tensor(item["input_values"]).unsqueeze(0)
        audio_length = audio.shape[1]

        frame_counts = frame_counts[:audio_length]
        frame_keypoints = torch.nonzero(frame_counts.diff() > 0).squeeze(1) + 1

        attention_mask = torch.zeros(batch_size, audio_length, dtype=torch.int32).to(model.device)

        # NB we start from 1 here.
        # We want the output at frame i to have input sufficient for computing up and through
        # frame i. So we should map output frame i to input keypoint i+1.
        for i in range(1, frame_keypoints.shape[0], batch_size):
            batch_frame_targets = torch.arange(i, min(i + batch_size, frame_keypoints.shape[0]))
            batch_keypoints = frame_keypoints[i:i + batch_size]
            batch_length = max(batch_keypoints)
            real_batch_size = batch_keypoints.shape[0]

            batch_inputs = audio[:, :batch_length].to(model.device)
            batch_inputs = torch.tile(batch_inputs, (real_batch_size, 1))
            for j, frame_keypoint in enumerate(batch_keypoints):
                batch_inputs[j, frame_keypoint:] = 0

            attention_mask.fill_(0)
            for j, frame_keypoint in enumerate(batch_keypoints):
                attention_mask[j, :frame_keypoint] = 1

            with torch.no_grad():
                output = model(output_hidden_states=True,
                              input_values=batch_inputs,
                              attention_mask=attention_mask[:real_batch_size, :batch_length])

            layer_states = {
                layer: output.hidden_states[layer][torch.arange(real_batch_size), batch_frame_targets - 1].cpu()
                for layer in layers}
            writer.write(layer_states,
                         [(idx, j - 1) for j in range(i, i + real_batch_size)],
                         {idx: frame_counts[-1].item() / audio_length})

    # Extract and un-pad hidden representations from the model
    if pseudo_causal:
        max_length = max(pa.compute.list_value_length(dataset._data["input_values"]).to_pylist())

        # Find the input wav frames at which a new frame is created.
        # pre-calculate and share across the invocations of extract_representations_pseudo_causal
        frame_counts = model._get_feat_extract_output_lengths(torch.arange(0, max_length))

        dataset.map(extract_representations_pseudo_causal,
                    fn_kwargs={"frame_counts": frame_counts, "max_length": max_length},
                    batched=False,
                    with_indices=True,
                    desc="Extracting hidden states")
    else:
        dataset.map(extract_representations,
                    batched=True, batch_size=batch_size,
                    with_indices=True,
                    desc="Extracting hidden states")


def get_layer_paths(config: DictConfig) -> dict[int, str]:
    """
    Layers to extract and their output paths.

    By default extracts `base_model.layer` to `base_model.hidden_state_path`. Set
    `+base_model.extract_layers=[0,1,...]` to extract several layers from one
    forward pass; `hidden_state_path` must then contain a `{layer}` placeholder.
    """
    layers: Optional[list[int]] = config.base_model.get("extract_layers")
    path_template = str(config.base_model.hidden_state_path)
    if layers is None:
        return {int(config.base_model.layer): path_template}

    if "{layer}" not in path_template:
        raise ValueError("hidden_state_path must contain {layer} when extract_layers is set")
    return {int(layer): path_template.format(layer=layer) for layer in layers}


def truncate_encoder(model: transformers.PreTrainedModel, top_layer: int) -> None:
    """
    Drop the transformer layers above `top_layer`, so that a forward pass stops there.

    `output.hidden_states[L]` is the output of layer L (0 = the encoder input), which does not
    depend on the layers above it, so the extracted states are unchanged; extracting layer 8
    alone then skips a third of the transformer. Opt in with
    `+base_model.truncate_above_extracted=true`.

    Only for post-LN encoders (wav2vec2-base: `do_stable_layer_norm=False`). Pre-LN encoders
    apply a final layer norm to the last hidden state, which truncation would move.
    """
    if getattr(model.config, "do_stable_layer_norm", False):
        raise ValueError("truncate_above_extracted is only valid for post-LN encoders "
                         "(do_stable_layer_norm=False)")
    num_layers = len(model.encoder.layers)
    if top_layer < num_layers:
        model.encoder.layers = model.encoder.layers[:top_layer]
        print(f"truncate_above_extracted: running {top_layer} of {num_layers} transformer layers")


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    feature_extractor = prepare_feature_extractor(config)
    dataset = datasets.load_from_disk(config.dataset.processed_data_dir).with_format("torch")

    model = transformers.AutoModel.from_pretrained(config.base_model.model_ref).to(config.device)

    layer_paths = get_layer_paths(config)
    if config.base_model.get("truncate_above_extracted", False):
        truncate_encoder(model, max(layer_paths))
    writer = HiddenStateWriter(layer_paths, model.name_or_path, model.config.hidden_size)
    try:
        extract_hidden_states(
            dataset, model, feature_extractor, writer, list(layer_paths.keys()),
            pseudo_causal=config.base_model.pseudo_causal)
    except BaseException:
        writer.close(complete=False)
        raise
    writer.close()


if __name__ == "__main__":
    main()
