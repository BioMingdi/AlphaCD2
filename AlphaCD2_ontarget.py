#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import pickle
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

            fields = line.split("\t") if "\t" in line else line.split()

            if len(fields) < 2:
                raise ValueError(
                    f"Line {line_number}: expected at least two columns "
                    "(name and sequence)."
                )

            name = fields[0].strip()
            sequence_raw = fields[1].strip()

            if has_header and not rows:
                if (
                    name.lower() in {"name", "id", "seq_id", "sequence_id"}
                    and sequence_raw.lower()
                    in {"sequence", "seq", "protein_sequence"}
                ):
                    continue

            sequence = clean_sequence(sequence_raw)

            if not name:
                raise ValueError(
                    f"Line {line_number}: empty sequence name."
                )

            if not sequence:
                raise ValueError(
                    f"Line {line_number}: empty protein sequence."
                )

            rows.append(
                {
                    "name": name,
                    "sequence": sequence,
                }
            )

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise ValueError(
            f"No valid sequences were read from {path}."
        )

    duplicated = frame.loc[
        frame["name"].duplicated(),
        "name",
    ].tolist()

    if duplicated:
        raise ValueError(
            "Sequence names must be unique. Duplicate examples: "
            + ", ".join(duplicated[:10])
        )

    return frame


def resolve_device(
    device: str | torch.device,
    label: str = "Device",
) -> torch.device:
    device = torch.device(device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            print(
                f"[{label}] CUDA is unavailable; falling back to CPU.",
                flush=True,
            )
            return torch.device("cpu")

        if device.index is None:
            return torch.device(
                f"cuda:{torch.cuda.current_device()}"
            )

        return device

    if device.type == "cpu":
        return device

    raise RuntimeError(
        f"{label}: unsupported device '{device}'. "
        "Use 'cuda', 'cuda:N', or 'cpu'."
    )


def unwrap_state_dict(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError(
            "The ESM-C checkpoint is not a dictionary/state_dict. "
            f"Found: {type(obj)!r}"
        )

    candidate = obj

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
        "module",
    ):
        value = candidate.get(key)

        if isinstance(value, dict) and value:
            candidate = value
            break

    if not candidate:
        raise ValueError(
            "The ESM-C checkpoint contains an empty state_dict."
        )

    return candidate


def strip_uniform_prefix(
    state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    keys = list(state_dict)

    for prefix in (
        "module.",
        "_orig_mod.",
        "model.",
    ):
        if (
            keys
            and sum(key.startswith(prefix) for key in keys)
            / len(keys)
            >= 0.95
        ):
            return {
                (
                    key[len(prefix):]
                    if key.startswith(prefix)
                    else key
                ): value
                for key, value in state_dict.items()
            }

    return state_dict


def load_esmc_state_dict(
    path: Path,
) -> Dict[str, Any]:
    try:
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        obj = torch.load(
            path,
            map_location="cpu",
        )

    return strip_uniform_prefix(
        unwrap_state_dict(obj)
    )


def build_local_esmc(
    checkpoint: Path,
    device: torch.device,
    use_flash_attn: bool,
):
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers

    device = resolve_device(
        device,
        "ESM-C",
    )

    if device.type == "cuda":
        model_dtype = torch.bfloat16
        flash_attn = bool(use_flash_attn)

        print(
            f"[ESM-C] Device: {device}; "
            f"CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}",
            flush=True,
        )
    else:
        model_dtype = torch.float32
        flash_attn = False

        print(
            "[ESM-C] Device: CPU; "
            "FlashAttention disabled.",
            flush=True,
        )

    kwargs = {
        "d_model": 1152,
        "n_heads": 18,
        "n_layers": 36,
        "tokenizer": get_esmc_model_tokenizers(),
        "use_flash_attn": flash_attn,
    }

    print(
        f"[ESM-C] Constructing model on {device} "
        f"with dtype={model_dtype}.",
        flush=True,
    )

    previous_dtype = torch.get_default_dtype()

    try:
        torch.set_default_dtype(model_dtype)

        with torch.device(device):
            model = ESMC(**kwargs).eval()

    except Exception as exc:
        raise RuntimeError(
            f"Failed to construct ESM-C on {device}: {exc}"
        ) from exc

    finally:
        torch.set_default_dtype(previous_dtype)

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
            "ESM-C contains unmaterialized meta tensors.\n"
            f"Parameters: {meta_parameters[:20]}\n"
            f"Buffers: {meta_buffers[:20]}"
        )

    wrong_parameters = [
        (name, str(parameter.device))
        for name, parameter in model.named_parameters()
        if parameter.device != device
    ]

    wrong_buffers = [
        (name, str(buffer.device))
        for name, buffer in model.named_buffers()
        if buffer.device != device
    ]

    if wrong_parameters or wrong_buffers:
        raise RuntimeError(
            "Some ESM-C tensors are on the wrong device.\n"
            f"Parameters: {wrong_parameters[:20]}\n"
            f"Buffers: {wrong_buffers[:20]}"
        )

    state_dict = load_esmc_state_dict(
        checkpoint
    )

    model.load_state_dict(
        state_dict,
        strict=True,
        assign=False,
    )

    del state_dict

    model.eval()

    first_parameter = next(
        model.parameters()
    )

    print(
        f"[ESM-C] Model ready: "
        f"device={first_parameter.device}, "
        f"dtype={first_parameter.dtype}",
        flush=True,
    )

    return model


@torch.inference_mode()
def generate_raw_embedding(
    client,
    sequence: str,
    device: torch.device,
) -> torch.Tensor:
    from esm.sdk.api import ESMProteinTensor, LogitsConfig
    from esm.tokenization import EsmSequenceTokenizer

    device = resolve_device(
        device,
        "ESM-C",
    )

    tokenizer = EsmSequenceTokenizer()
    token_ids = tokenizer.encode(
        sequence
    )

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

    if not isinstance(
        embedding,
        torch.Tensor,
    ):
        embedding = torch.as_tensor(
            embedding
        )

    return embedding.detach().to(
        device="cpu",
        dtype=torch.float32,
    )


def load_cache(
    path: Path,
) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}

    with path.open("rb") as handle:
        cache = pickle.load(handle)

    if not isinstance(cache, dict):
        raise TypeError(
            f"Embedding cache is not a dictionary: {path}"
        )

    return cache


def save_cache(
    cache: Dict[str, Dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("wb") as handle:
        pickle.dump(
            cache,
            handle,
        )


def update_embedding_cache(
    frame: pd.DataFrame,
    cache_path: Path,
    checkpoint: Path,
    esm_device: torch.device,
    use_flash_attn: bool,
    regenerate: bool,
) -> Dict[str, Dict[str, Any]]:
    esm_device = resolve_device(
        esm_device,
        "ESM-C",
    )

    if regenerate and cache_path.exists():
        cache_path.unlink()

    cache = load_cache(
        cache_path
    )

    missing_rows = []

    for row in frame.itertuples(
        index=False
    ):
        record = cache.get(
            row.name
        )

        if record is None:
            missing_rows.append(
                row
            )
            continue

        cached_sequence = clean_sequence(
            record.get(
                "sequence",
                "",
            )
        )

        if cached_sequence != row.sequence:
            raise ValueError(
                f"Cached sequence for '{row.name}' differs "
                "from the current input. "
                "Use --regenerate-embeddings or a new cache file."
            )

        if "embedding" not in record:
            missing_rows.append(
                row
            )

    if not missing_rows:
        print(
            f"Embedding cache contains all "
            f"{len(frame)} sequences: {cache_path}",
            flush=True,
        )
        return cache

    print(
        f"Loading ESM-C checkpoint:\n"
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

    for index, row in enumerate(
        missing_rows,
        start=1,
    ):
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
            f"[{index}/{len(missing_rows)}] "
            f"{row.name}: "
            f"sequence_length={len(row.sequence)}, "
            f"raw_embedding_shape="
            f"{tuple(embedding.shape)}",
            flush=True,
        )

        save_cache(
            cache,
            cache_path,
        )

        if esm_device.type == "cuda":
            torch.cuda.empty_cache()

    del client

    if esm_device.type == "cuda":
        torch.cuda.empty_cache()

    return cache


def features_from_cache(
    frame: pd.DataFrame,
    cache: Dict[str, Dict[str, Any]],
    target_dim: int,
) -> Tuple[np.ndarray, List[str]]:
    features = []
    shapes = []

    for row in frame.itertuples(
        index=False
    ):
        if row.name not in cache:
            raise KeyError(
                f"Sequence '{row.name}' is missing "
                "from the embedding cache."
            )

        record = cache[
            row.name
        ]

        if "embedding" not in record:
            raise KeyError(
                f"Embedding for '{row.name}' "
                "is missing from the cache."
            )

        embedding = record[
            "embedding"
        ]

        if isinstance(
            embedding,
            torch.Tensor,
        ):
            embedding = (
                embedding
                .detach()
                .cpu()
                .numpy()
            )

        embedding = np.asarray(
            embedding
        )

        shapes.append(
            "x".join(
                map(
                    str,
                    embedding.shape,
                )
            )
        )

        flattened = embedding.reshape(
            -1
        )

        if flattened.size >= target_dim:
            flattened = flattened[
                :target_dim
            ]
        else:
            flattened = np.pad(
                flattened,
                (
                    0,
                    target_dim
                    - flattened.size,
                ),
                mode="constant",
            )

        features.append(
            flattened.astype(
                np.float32,
                copy=False,
            )
        )

    return (
        np.asarray(
            features,
            dtype=np.float32,
        ),
        shapes,
    )


@torch.inference_mode()
def predict_array(
    model,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    device = resolve_device(
        device,
        "Prediction",
    )

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(
                features
            )
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    predictions = []

    model.eval()

    for (batch_x,) in loader:
        batch_x = batch_x.to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        output, _ = model(
            batch_x
        )

        predictions.extend(
            output
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

    return np.asarray(
        predictions,
        dtype=float,
    )


def load_model_config(
    checkpoint: Dict[str, Any],
) -> ModelConfig:
    cfg_dict = dict(
        checkpoint[
            "model_config"
        ]
    )

    if "cnn_hidden_dims" in cfg_dict:
        cfg_dict[
            "cnn_hidden_dims"
        ] = tuple(
            cfg_dict[
                "cnn_hidden_dims"
            ]
        )

    if "fusion_dims" in cfg_dict:
        cfg_dict[
            "fusion_dims"
        ] = tuple(
            cfg_dict[
                "fusion_dims"
            ]
        )

    return ModelConfig(
        **cfg_dict
    )


def load_bilstm_checkpoint(
    path: Path,
) -> Dict[str, Any]:
    if (
        not path.is_file()
        or path.stat().st_size == 0
    ):
        raise FileNotFoundError(
            f"BiLSTM checkpoint is missing or empty: {path}"
        )

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def read_manifest(
    manifest_path: Path,
) -> Dict[str, Any]:
    if (
        not manifest_path.is_file()
        or manifest_path.stat().st_size == 0
    ):
        raise FileNotFoundError(
            f"Manifest is missing or empty: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if manifest.get(
        "model_mode"
    ) != "bilstm":
        raise ValueError(
            f"Manifest is not a BiLSTM manifest: "
            f"{manifest_path}"
        )

    entries = manifest.get(
        "models",
        [],
    )

    if not entries:
        raise ValueError(
            "The BiLSTM manifest contains no model entries."
        )

    return manifest


def read_target_dim_from_manifest(
    manifest_path: Path,
) -> int:
    manifest = read_manifest(
        manifest_path
    )

    entry = manifest[
        "models"
    ][0]

    checkpoint_path = (
        manifest_path.parent
        / entry["checkpoint"]
    )

    checkpoint = load_bilstm_checkpoint(
        checkpoint_path
    )

    return int(
        checkpoint[
            "preprocessing"
        ][
            "target_dim"
        ]
    )


def predict_all_seeds(
    manifest_path: Path,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], int]:
    device = resolve_device(
        device,
        "Prediction",
    )

    manifest = read_manifest(
        manifest_path
    )

    entries = manifest[
        "models"
    ]

    seed_predictions: Dict[
        int,
        np.ndarray,
    ] = {}

    target_dim = None

    for entry in entries:
        checkpoint_path = (
            manifest_path.parent
            / entry["checkpoint"]
        )

        scaler_path = (
            manifest_path.parent
            / entry["scaler"]
        )

        if (
            not scaler_path.is_file()
            or scaler_path.stat().st_size == 0
        ):
            raise FileNotFoundError(
                f"Scaler is missing or empty: {scaler_path}"
            )

        checkpoint = load_bilstm_checkpoint(
            checkpoint_path
        )

        current_target_dim = int(
            checkpoint[
                "preprocessing"
            ][
                "target_dim"
            ]
        )

        if target_dim is None:
            target_dim = (
                current_target_dim
            )

        elif (
            current_target_dim
            != target_dim
        ):
            raise ValueError(
                "Target dimension differs "
                "among seed checkpoints."
            )

        if features.shape[1] != current_target_dim:
            raise ValueError(
                f"Feature dimension "
                f"{features.shape[1]} does not match "
                f"checkpoint target_dim "
                f"{current_target_dim}."
            )

        cfg = load_model_config(
            checkpoint
        )

        model = AlphaCD2Regressor(
            cfg
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        model.to(
            device
        )

        with scaler_path.open(
            "rb"
        ) as handle:
            scaler = pickle.load(
                handle
            )

        scaled = scaler.transform(
            features
        ).astype(
            np.float32
        )

        prediction = predict_array(
            model=model,
            features=scaled,
            batch_size=batch_size,
            device=device,
        )

        seed = int(
            checkpoint[
                "training"
            ][
                "seed"
            ]
        )

        if seed in seed_predictions:
            raise ValueError(
                f"Duplicate seed {seed} "
                "in BiLSTM manifest."
            )

        seed_predictions[
            seed
        ] = prediction

        print(
            f"BiLSTM seed={seed}: "
            f"prediction range="
            f"[{prediction.min():.6f}, "
            f"{prediction.max():.6f}]",
            flush=True,
        )

        del model
        del checkpoint

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if target_dim is None:
        raise RuntimeError(
            "Unable to determine target dimension."
        )

    return (
        seed_predictions,
        target_dim,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict AlphaCD2 on-target activity "
            "from protein sequences."
        )
    )

    parser.add_argument(
        "--input-txt",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--esm-checkpoint",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--embedding-cache-pkl",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--has-header",
        action="store_true",
    )

    parser.add_argument(
        "--regenerate-embeddings",
        action="store_true",
    )

    parser.add_argument(
        "--esm-device",
        default="cuda",
    )

    parser.add_argument(
        "--predict-device",
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--disable-flash-attn",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    os.environ[
        "HF_HUB_OFFLINE"
    ] = "1"

    os.environ[
        "TRANSFORMERS_OFFLINE"
    ] = "1"

    os.environ[
        "HF_DATASETS_OFFLINE"
    ] = "1"

    for required in (
        args.input_txt,
        args.manifest,
        args.esm_checkpoint,
    ):
        if (
            not required.is_file()
            or required.stat().st_size == 0
        ):
            raise FileNotFoundError(
                f"Required file is missing or empty: "
                f"{required}"
            )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be at least 1."
        )

    frame = read_name_sequence(
        args.input_txt,
        args.has_header,
    )

    print(
        f"Read {len(frame)} sequences "
        f"from {args.input_txt}",
        flush=True,
    )

    esm_device = resolve_device(
        args.esm_device,
        "ESM-C",
    )

    predict_device = resolve_device(
        args.predict_device,
        "Prediction",
    )

    print(
        f"ESM-C device: {esm_device}",
        flush=True,
    )

    print(
        f"Prediction device: {predict_device}",
        flush=True,
    )

    cache_path = (
        args.embedding_cache_pkl
        if args.embedding_cache_pkl
        is not None
        else args.output.with_suffix(
            ".embeddings.pkl"
        )
    )

    cache = update_embedding_cache(
        frame=frame,
        cache_path=cache_path,
        checkpoint=args.esm_checkpoint,
        esm_device=esm_device,
        use_flash_attn=(
            not args.disable_flash_attn
        ),
        regenerate=(
            args.regenerate_embeddings
        ),
    )

    target_dim = (
        read_target_dim_from_manifest(
            args.manifest
        )
    )

    features, embedding_shapes = (
        features_from_cache(
            frame=frame,
            cache=cache,
            target_dim=target_dim,
        )
    )

    (
        seed_predictions,
        verified_target_dim,
    ) = predict_all_seeds(
        manifest_path=args.manifest,
        features=features,
        batch_size=args.batch_size,
        device=predict_device,
    )

    if (
        verified_target_dim
        != target_dim
    ):
        raise RuntimeError(
            "Inconsistent target dimensions."
        )

    result = frame.copy()

    result[
        "sequence_length"
    ] = result[
        "sequence"
    ].str.len()

    result[
        "raw_embedding_shape"
    ] = embedding_shapes

    ordered_seeds = sorted(
        seed_predictions
    )

    for seed in ordered_seeds:
        result[
            f"prediction_seed_{seed}"
        ] = seed_predictions[
            seed
        ]

    matrix = np.column_stack(
        [
            seed_predictions[
                seed
            ]
            for seed in ordered_seeds
        ]
    )

    result[
        "predicted_efficiency"
    ] = matrix.mean(
        axis=1
    )

    result[
        "predicted_efficiency_percent"
    ] = (
        result[
            "predicted_efficiency"
        ]
        * 100.0
    )

    if matrix.shape[1] > 1:
        prediction_sd = matrix.std(
            axis=1,
            ddof=1,
        )
    else:
        prediction_sd = np.zeros(
            len(result),
            dtype=float,
        )

    result[
        "prediction_sd"
    ] = prediction_sd

    result[
        "prediction_sd_percent"
    ] = (
        prediction_sd
        * 100.0
    )

    result[
        "prediction_min"
    ] = matrix.min(
        axis=1
    )

    result[
        "prediction_max"
    ] = matrix.max(
        axis=1
    )

    prediction_rank = (
        result[
            "predicted_efficiency"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    result.insert(
        0,
        "prediction_rank",
        prediction_rank,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        args.output,
        sep="\t",
        index=False,
    )

    summary = {
        "input_txt": str(
            args.input_txt.resolve()
        ),
        "manifest": str(
            args.manifest.resolve()
        ),
        "esm_checkpoint": str(
            args.esm_checkpoint.resolve()
        ),
        "embedding_cache_pkl": str(
            cache_path.resolve()
        ),
        "output": str(
            args.output.resolve()
        ),
        "esm_device": str(
            esm_device
        ),
        "predict_device": str(
            predict_device
        ),
        "n_sequences": int(
            len(result)
        ),
        "n_seed_models": int(
            len(ordered_seeds)
        ),
        "seeds": [
            int(seed)
            for seed in ordered_seeds
        ],
        "target_dim": int(
            target_dim
        ),
        "prediction_mean": float(
            result[
                "predicted_efficiency"
            ].mean()
        ),
        "prediction_min": float(
            result[
                "predicted_efficiency"
            ].min()
        ),
        "prediction_max": float(
            result[
                "predicted_efficiency"
            ].max()
        ),
        "mean_between_seed_sd": float(
            result[
                "prediction_sd"
            ].mean()
        ),
    }

    summary_path = Path(
        str(args.output)
        + ".summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
        ),
        flush=True,
    )

    print(
        f"Saved predictions: "
        f"{args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
