#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict AlphaCD2 activity directly from a two-column name/sequence TXT file.

Input format
------------
No header:
    NeoCD01<TAB>MAEKKK...
    NeoCD02<TAB>MNNRRL...

With header:
    name<TAB>sequence
    NeoCD01<TAB>MAEKKK...

Workflow
--------
1. Read and clean protein sequences.
2. Generate ESM-C 600M embeddings with a local checkpoint.
3. Preserve the legacy raw ESM-C embedding convention used for training.
4. Reproduce embedding.flatten()[:1152] preprocessing.
5. Predict with all full-data BiLSTM seeds from final_bilstm_manifest.json.
6. Report the seed mean, standard deviation, range and activity percentage.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from alphacd2_seed42_model import AlphaCD2Regressor, ModelConfig


VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")


def clean_sequence(sequence: str) -> str:
    sequence = "".join(str(sequence).upper().split())
    return "".join(aa if aa in VALID_AA else "X" for aa in sequence)


def read_name_sequence(path: Path, has_header: bool) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "\t" in line:
                fields = line.split("\t")
            else:
                fields = line.split()

            if len(fields) < 2:
                raise ValueError(
                    f"Line {line_number}: expected at least two columns "
                    "(name and sequence)."
                )

            name = fields[0].strip()
            sequence = clean_sequence(fields[1])

            if has_header and not rows:
                header_name = name.lower()
                header_sequence = fields[1].strip().lower()
                if header_name in {"name", "id", "seq_id", "sequence_id"}:
                    if header_sequence in {"sequence", "seq", "protein_sequence"}:
                        continue

            if not name:
                raise ValueError(f"Line {line_number}: empty sequence name.")
            if not sequence:
                raise ValueError(f"Line {line_number}: empty protein sequence.")

            rows.append(
                {
                    "name": name,
                    "sequence": sequence,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No valid sequences were read from {path}.")

    duplicated = frame.loc[frame["name"].duplicated(), "name"].tolist()
    if duplicated:
        raise ValueError(
            "Sequence names must be unique. Duplicate examples: "
            + ", ".join(duplicated[:10])
        )

    return frame


def unwrap_state_dict(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError(
            "The ESM-C checkpoint is not a dictionary/state_dict. "
            f"Found: {type(obj)!r}"
        )

    candidate = obj
    for key in ("state_dict", "model_state_dict", "model", "module"):
        value = candidate.get(key)
        if isinstance(value, dict) and value:
            candidate = value
            break

    if not candidate:
        raise ValueError("The ESM-C checkpoint contains an empty state_dict.")

    return candidate


def strip_uniform_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    keys = list(state_dict)
    for prefix in ("module.", "_orig_mod.", "model."):
        if keys and sum(key.startswith(prefix) for key in keys) / len(keys) >= 0.95:
            return {
                key[len(prefix):] if key.startswith(prefix) else key: value
                for key, value in state_dict.items()
            }
    return state_dict


def load_esmc_state_dict(path: Path) -> Dict[str, Any]:
    # The checkpoint is first read on CPU only as serialized storage. The ESM-C
    # module itself is never instantiated on CPU. Its tensors are materialized
    # directly on the requested GPU below.
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    return strip_uniform_prefix(unwrap_state_dict(obj))


def move_state_dict_to_device(
    state_dict: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            moved[key] = value
            continue

        if device.type == "cuda" and value.is_floating_point():
            moved[key] = value.to(
                device=device,
                dtype=torch.bfloat16,
                non_blocking=False,
            )
        else:
            moved[key] = value.to(device=device, non_blocking=False)
    return moved


def build_local_esmc(
    checkpoint: Path,
    device: torch.device,
    use_flash_attn: bool,
):
    """Construct ESM-C directly on GPU without leaving meta tensors behind.

    The previous meta-device + load_state_dict(assign=True) approach correctly
    materialized checkpoint parameters, but ESM-C also contains non-persistent
    buffers such as rotary-position inv_freq. Those buffers are not present in
    the checkpoint and therefore remained on the meta device, causing:

        RuntimeError: Tensor on device meta is not on the expected device cuda:0

    This implementation creates the complete module directly on the requested
    GPU in bfloat16. Consequently, parameters and non-persistent buffers are
    initialized on CUDA before the CPU checkpoint is copied into the module.
    """
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers

    device = torch.device(device)

    if device.type != "cuda":
        raise RuntimeError(
            "GPU construction was requested, but the resolved ESM-C device "
            f"is {device}. Check CUDA_VISIBLE_DEVICES and --esm-device."
        )

    # Normalize an unspecified CUDA device such as "cuda" to the logical CUDA
    # device visible inside the current process. For example,
    # CUDA_VISIBLE_DEVICES=1 maps physical GPU 1 to logical cuda:0.
    if device.index is None:
        device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
        )

    print(
        f"[ESM-C] Resolved logical CUDA device: {device}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )

    kwargs = {
        "d_model": 1152,
        "n_heads": 18,
        "n_layers": 36,
        "tokenizer": get_esmc_model_tokenizers(),
        "use_flash_attn": bool(use_flash_attn),
    }

    print(
        f"[ESM-C] Constructing the complete model directly on {device} "
        "in bfloat16 (no meta-device parameters or buffers).",
        flush=True,
    )

    # Construct every parameter and buffer directly on the visible GPU.
    # Temporarily changing the default floating dtype avoids first allocating a
    # full float32 model and then creating a second bfloat16 copy.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(device):
            model = ESMC(**kwargs).eval()
    except Exception as exc:
        raise RuntimeError(
            f"Direct CUDA construction of ESM-C failed on {device}: {exc}"
        ) from exc
    finally:
        torch.set_default_dtype(previous_dtype)

    # Verify that no parameter or buffer was left on the meta device.
    meta_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_meta
    ]
    meta_buffers = [
        name
        for name, buffer in model.named_buffers()
        if buffer.is_meta
    ]
    if meta_parameters or meta_buffers:
        raise RuntimeError(
            "ESM-C direct CUDA construction unexpectedly left meta tensors.\n"
            f"Meta parameters ({len(meta_parameters)}): "
            f"{meta_parameters[:20]}\n"
            f"Meta buffers ({len(meta_buffers)}): {meta_buffers[:20]}"
        )

    wrong_device_parameters = [
        (name, str(parameter.device))
        for name, parameter in model.named_parameters()
        if parameter.device != device
    ]
    wrong_device_buffers = [
        (name, str(buffer.device))
        for name, buffer in model.named_buffers()
        if buffer.device != device
    ]
    if wrong_device_parameters or wrong_device_buffers:
        raise RuntimeError(
            "Some ESM-C tensors were not constructed on the requested GPU.\n"
            f"Parameters: {wrong_device_parameters[:20]}\n"
            f"Buffers: {wrong_device_buffers[:20]}"
        )

    # Keep checkpoint storage on CPU. load_state_dict copies and casts values
    # into the already materialized CUDA tensors, avoiding a second full GPU
    # state-dict copy.
    cpu_state_dict = load_esmc_state_dict(checkpoint)
    incompatible = model.load_state_dict(
        cpu_state_dict,
        strict=True,
        assign=False,
    )

    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(
            "Local ESM-C checkpoint does not match ESM-C 600M.\n"
            f"Missing ({len(missing)}): {missing[:20]}\n"
            f"Unexpected ({len(unexpected)}): {unexpected[:20]}"
        )

    del cpu_state_dict
    model.eval()

    first_parameter = next(model.parameters())
    print(
        f"[ESM-C] Model ready: device={first_parameter.device}, "
        f"dtype={first_parameter.dtype}; "
        f"meta_parameters=0, meta_buffers=0",
        flush=True,
    )
    return model


@torch.inference_mode()
def generate_raw_embedding(client, sequence: str, device: torch.device) -> torch.Tensor:
    """Match the original legacy embedding-generation API and tensor convention."""
    from esm.sdk.api import ESMProteinTensor, LogitsConfig
    from esm.tokenization import EsmSequenceTokenizer

    tokenizer = EsmSequenceTokenizer()
    token_ids = tokenizer.encode(sequence)

    protein_tensor = ESMProteinTensor(
        sequence=torch.tensor(
            token_ids,
            dtype=torch.long,
            device=device,
        )
    )
    result = client.logits(
        protein_tensor,
        LogitsConfig(
            sequence=True,
            return_embeddings=True,
        ),
    )

    embedding = result.embeddings
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.as_tensor(embedding)

    # Keep the raw shape exactly as returned by ESM-C. The legacy training
    # pipeline flattened this tensor before truncating/padding to 1152.
    return embedding.detach().to(device="cpu", dtype=torch.float32)


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, dict):
        raise TypeError(f"Embedding cache is not a dictionary: {path}")
    return cache


def update_embedding_cache(
    frame: pd.DataFrame,
    cache_path: Path,
    checkpoint: Path,
    esm_device: torch.device,
    use_flash_attn: bool,
    regenerate: bool,
) -> Dict[str, Dict[str, Any]]:
    if regenerate and cache_path.exists():
        cache_path.unlink()

    cache = load_cache(cache_path)

    missing_rows = []
    for row in frame.itertuples(index=False):
        record = cache.get(row.name)
        if record is None:
            missing_rows.append(row)
            continue

        cached_sequence = clean_sequence(record.get("sequence", ""))
        if cached_sequence != row.sequence:
            raise ValueError(
                f"Cached sequence for '{row.name}' differs from the current input. "
                "Use --regenerate-embeddings or a new cache file."
            )
        if "embedding" not in record:
            missing_rows.append(row)

    if not missing_rows:
        print(
            f"Embedding cache already contains all {len(frame)} sequences: "
            f"{cache_path}",
            flush=True,
        )
        return cache

    print(
        f"Loading local ESM-C 600M checkpoint:\n"
        f"  {checkpoint}\n"
        f"Embedding device: {esm_device}\n"
        f"Sequences to embed: {len(missing_rows)}",
        flush=True,
    )

    client = build_local_esmc(
        checkpoint=checkpoint,
        device=esm_device,
        use_flash_attn=use_flash_attn,
    )

    for index, row in enumerate(missing_rows, start=1):
        embedding = generate_raw_embedding(
            client,
            row.sequence,
            esm_device,
        )
        cache[row.name] = {
            "sequence": row.sequence,
            "embedding": embedding,
        }

        print(
            f"[{index}/{len(missing_rows)}] {row.name}: "
            f"sequence_length={len(row.sequence)}, "
            f"raw_embedding_shape={tuple(embedding.shape)}",
            flush=True,
        )

        # Persist after every sequence so a long run can resume safely.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(cache, handle)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del client
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cache


def features_from_cache(
    frame: pd.DataFrame,
    cache: Dict[str, Dict[str, Any]],
    target_dim: int,
) -> Tuple[np.ndarray, List[str]]:
    features = []
    shapes = []

    for row in frame.itertuples(index=False):
        record = cache[row.name]
        embedding = record["embedding"]

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        embedding = np.asarray(embedding)

        shapes.append("x".join(map(str, embedding.shape)))

        flattened = embedding.flatten()
        if flattened.size >= target_dim:
            flattened = flattened[:target_dim]
        else:
            flattened = np.pad(
                flattened,
                (0, target_dim - flattened.size),
                mode="constant",
            )

        features.append(flattened.astype(np.float32))

    return np.asarray(features, dtype=np.float32), shapes


@torch.inference_mode()
def predict_array(
    model,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
    )

    predictions = []
    model.eval()
    for (batch_x,) in loader:
        output, _ = model(batch_x.to(device, non_blocking=True))
        predictions.extend(
            output.detach().cpu().numpy().reshape(-1)
        )

    return np.asarray(predictions, dtype=float)


def load_model_config(checkpoint: Dict[str, Any]) -> ModelConfig:
    cfg_dict = dict(checkpoint["model_config"])
    if "cnn_hidden_dims" in cfg_dict:
        cfg_dict["cnn_hidden_dims"] = tuple(cfg_dict["cnn_hidden_dims"])
    if "fusion_dims" in cfg_dict:
        cfg_dict["fusion_dims"] = tuple(cfg_dict["fusion_dims"])
    return ModelConfig(**cfg_dict)


def predict_all_seeds(
    manifest_path: Path,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_mode") != "bilstm":
        raise ValueError(
            f"Manifest is not a BiLSTM manifest: {manifest_path}"
        )

    entries = manifest.get("models", [])
    if not entries:
        raise ValueError("The BiLSTM manifest contains no model entries.")

    seed_predictions: Dict[int, np.ndarray] = {}
    target_dim = None

    for entry in entries:
        checkpoint_path = manifest_path.parent / entry["checkpoint"]
        scaler_path = manifest_path.parent / entry["scaler"]

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
            )

        current_target_dim = int(
            checkpoint["preprocessing"]["target_dim"]
        )
        if target_dim is None:
            target_dim = current_target_dim
        elif current_target_dim != target_dim:
            raise ValueError(
                "Target dimension differs among seed checkpoints."
            )

        cfg = load_model_config(checkpoint)
        model = AlphaCD2Regressor(cfg)
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        model.to(device)

        with scaler_path.open("rb") as handle:
            scaler = pickle.load(handle)

        scaled = scaler.transform(features).astype(np.float32)
        prediction = predict_array(
            model,
            scaled,
            batch_size,
            device,
        )

        seed = int(checkpoint["training"]["seed"])
        seed_predictions[seed] = prediction

        print(
            f"BiLSTM seed={seed}: "
            f"prediction range=[{prediction.min():.6f}, "
            f"{prediction.max():.6f}]",
            flush=True,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert target_dim is not None
    return seed_predictions, target_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-txt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--esm-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--embedding-cache-pkl", type=Path)
    parser.add_argument("--has-header", action="store_true")
    parser.add_argument("--regenerate-embeddings", action="store_true")
    parser.add_argument("--esm-device", default="cuda")
    parser.add_argument("--predict-device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--disable-flash-attn",
        action="store_true",
    )
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    for required in (
        args.input_txt,
        args.manifest,
        args.esm_checkpoint,
    ):
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(
                f"Required file is missing or empty: {required}"
            )

    frame = read_name_sequence(
        args.input_txt,
        args.has_header,
    )
    print(f"Read {len(frame)} sequences from {args.input_txt}", flush=True)

    cache_path = args.embedding_cache_pkl
    if cache_path is None:
        cache_path = args.output.with_suffix(".embeddings.pkl")

    esm_device = torch.device(
        args.esm_device
        if args.esm_device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    predict_device = torch.device(
        args.predict_device
        if args.predict_device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )

    cache = update_embedding_cache(
        frame=frame,
        cache_path=cache_path,
        checkpoint=args.esm_checkpoint,
        esm_device=esm_device,
        use_flash_attn=not args.disable_flash_attn,
        regenerate=args.regenerate_embeddings,
    )

    # Read target_dim before constructing features.
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest.get("models", [])
    if not entries:
        raise ValueError("Manifest contains no BiLSTM model entries.")

    first_checkpoint_path = (
        args.manifest.parent / entries[0]["checkpoint"]
    )
    try:
        first_checkpoint = torch.load(
            first_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        first_checkpoint = torch.load(
            first_checkpoint_path,
            map_location="cpu",
        )

    target_dim = int(
        first_checkpoint["preprocessing"]["target_dim"]
    )

    features, embedding_shapes = features_from_cache(
        frame,
        cache,
        target_dim,
    )

    seed_predictions, verified_target_dim = predict_all_seeds(
        manifest_path=args.manifest,
        features=features,
        batch_size=args.batch_size,
        device=predict_device,
    )
    if verified_target_dim != target_dim:
        raise RuntimeError("Inconsistent target dimensions.")

    result = frame.copy()
    result["sequence_length"] = result["sequence"].str.len()
    result["raw_embedding_shape"] = embedding_shapes

    ordered_seeds = sorted(seed_predictions)
    for seed in ordered_seeds:
        result[f"prediction_seed_{seed}"] = seed_predictions[seed]

    matrix = np.column_stack(
        [seed_predictions[seed] for seed in ordered_seeds]
    )
    result["predicted_efficiency"] = matrix.mean(axis=1)
    result["predicted_efficiency_percent"] = (
        result["predicted_efficiency"] * 100.0
    )
    result["prediction_sd"] = (
        matrix.std(axis=1, ddof=1)
        if matrix.shape[1] > 1
        else 0.0
    )
    result["prediction_sd_percent"] = (
        result["prediction_sd"] * 100.0
    )
    result["prediction_min"] = matrix.min(axis=1)
    result["prediction_max"] = matrix.max(axis=1)

    descending_rank = (
        result["predicted_efficiency"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    result.insert(0, "prediction_rank", descending_rank)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        args.output,
        sep="\t",
        index=False,
    )

    summary = {
        "input_txt": str(args.input_txt.resolve()),
        "manifest": str(args.manifest.resolve()),
        "esm_checkpoint": str(args.esm_checkpoint.resolve()),
        "embedding_cache_pkl": str(cache_path.resolve()),
        "output": str(args.output.resolve()),
        "n_sequences": int(len(result)),
        "n_seed_models": int(len(ordered_seeds)),
        "seeds": ordered_seeds,
        "target_dim": int(target_dim),
        "prediction_mean": float(
            result["predicted_efficiency"].mean()
        ),
        "prediction_min": float(
            result["predicted_efficiency"].min()
        ),
        "prediction_max": float(
            result["predicted_efficiency"].max()
        ),
        "mean_between_seed_sd": float(
            result["prediction_sd"].mean()
        ),
    }

    summary_path = Path(str(args.output) + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved predictions: {args.output}", flush=True)


if __name__ == "__main__":
    main()
