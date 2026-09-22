#!/usr/bin/env python3
"""
Predict on-target efficiency and specificity from a two-column TXT file.

"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


MOTIF_CLASSES = ("AC", "CC", "GC", "TC")
WINDOW_MIN = 1
WINDOW_MAX = 14
WINDOW_POSITIONS = np.arange(WINDOW_MIN, WINDOW_MAX + 1)
N_WINDOW_POSITIONS = len(WINDOW_POSITIONS)


def load_ontarget_module(path: Path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"On-target prediction script not found: {path}")

    spec = importlib.util.spec_from_file_location(
        "alphacd2_final_ontarget_predictor",
        str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import on-target script: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = [
        "read_name_sequence",
        "update_embedding_cache",
        "features_from_cache",
        "predict_all_seeds",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            "The supplied on-target script is missing required functions: "
            + ", ".join(missing)
        )
    return module


def read_target_dim_from_manifest(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("models", [])
    if not entries:
        raise ValueError("Manifest contains no BiLSTM model entries.")

    checkpoint_path = manifest_path.parent / entries[0]["checkpoint"]
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

    return int(checkpoint["preprocessing"]["target_dim"])


def decode_contiguous_window(probabilities: np.ndarray, threshold: float = 0.5):
    p = np.asarray(probabilities, dtype=float)
    if p.shape[-1] != N_WINDOW_POSITIONS:
        raise ValueError(
            f"Expected {N_WINDOW_POSITIONS} window probabilities, got {p.shape}"
        )

    scores = p - float(threshold)
    if np.max(scores) <= 0:
        idx = int(np.argmax(p))
        pos = idx + WINDOW_MIN
        return pos, pos

    best_sum = -np.inf
    best_start = 0
    best_end = 0
    current_sum = 0.0
    current_start = 0

    for i, value in enumerate(scores):
        if current_sum <= 0:
            current_sum = float(value)
            current_start = i
        else:
            current_sum += float(value)

        current_len = i - current_start + 1
        best_len = best_end - best_start + 1

        if (
            current_sum > best_sum + 1e-12
            or (
                abs(current_sum - best_sum) <= 1e-12
                and current_len > best_len
            )
        ):
            best_sum = current_sum
            best_start = current_start
            best_end = i

    return best_start + WINDOW_MIN, best_end + WINDOW_MIN


@dataclass
class SpecificityConfig:
    input_dim: int = 1152
    bilstm_hidden_dim: int = 256
    bilstm_num_layers: int = 2
    adapter_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.30
    num_motif_classes: int = 4
    num_window_positions: int = N_WINDOW_POSITIONS


class SpecificityBackbone(nn.Module):
    def __init__(self, cfg: SpecificityConfig):
        super().__init__()
        h = cfg.bilstm_hidden_dim
        self.h = h

        self.sequence_creator = nn.Sequential(
            nn.Linear(cfg.input_dim, h * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.lstm = nn.LSTM(
            input_size=h,
            hidden_size=h,
            num_layers=cfg.bilstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.bilstm_num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(h * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.feature_extractor = nn.Sequential(
            nn.Linear(h * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        pseudo = self.sequence_creator(x).reshape(x.shape[0], 4, self.h)
        out, _ = self.lstm(pseudo)
        attn = torch.softmax(self.attention(out).squeeze(-1), dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * out, dim=1)
        return self.feature_extractor(pooled)


class TaskAdapter(nn.Module):
    def __init__(self, input_dim: int, dims: Tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        dim = input_dim
        for hidden in dims:
            layers.extend([
                nn.Linear(dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            dim = hidden
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = dim

    def forward(self, x):
        return self.net(x)


class ThreeTaskSpecificityA1WindowProfile(nn.Module):
    TASK_NAMES = ("motif", "window", "offtarget")

    def __init__(self, cfg: SpecificityConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = SpecificityBackbone(cfg)

        self.motif_adapter = TaskAdapter(128, cfg.adapter_dims, cfg.dropout)
        self.window_adapter = TaskAdapter(128, cfg.adapter_dims, cfg.dropout)
        self.offtarget_adapter = TaskAdapter(128, cfg.adapter_dims, cfg.dropout)

        task_dim = self.motif_adapter.output_dim

        self.motif_head = nn.Sequential(
            nn.Linear(task_dim, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout / 2),
            nn.Linear(128, cfg.num_motif_classes),
        )
        self.window_head = nn.Sequential(
            nn.Linear(task_dim, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout / 2),
            nn.Linear(128, cfg.num_window_positions),
        )
        self.offtarget_head = nn.Sequential(
            nn.Linear(task_dim, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout / 2),
            nn.Linear(128, 1),
        )

        self.task_log_vars = nn.Parameter(torch.zeros(len(self.TASK_NAMES)))

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)

        shared = self.backbone(x)
        return {
            "motif_logits": self.motif_head(self.motif_adapter(shared)),
            "window_logits": self.window_head(self.window_adapter(shared)),
            "offtarget_z": self.offtarget_head(
                self.offtarget_adapter(shared)
            ).squeeze(-1),
        }


def infer_specificity_scaler(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".input_scaler.pkl")


def load_specificity_model(checkpoint: Path, scaler_path: Path, device: torch.device):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Specificity checkpoint not found: {checkpoint}")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Specificity scaler not found: {scaler_path}")

    try:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")

    cfg_dict = dict(payload["model_config"])
    if "adapter_dims" in cfg_dict:
        cfg_dict["adapter_dims"] = tuple(cfg_dict["adapter_dims"])
    cfg = SpecificityConfig(**cfg_dict)

    model = ThreeTaskSpecificityA1WindowProfile(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()

    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)

    norm = payload["target_normalizer"]
    motif_classes = tuple(payload.get("motif_classes", MOTIF_CLASSES))
    metadata = payload.get("metadata", {})
    return model, scaler, norm, motif_classes, metadata, cfg


@torch.inference_mode()
def predict_specificity(
    model,
    scaler,
    norm,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
    window_threshold: float,
):
    scaled = scaler.transform(features).astype(np.float32)

    motif_probabilities = []
    window_probabilities = []
    offtarget_z = []

    for start in range(0, len(scaled), batch_size):
        xb = torch.from_numpy(scaled[start:start + batch_size]).to(
            device,
            non_blocking=True,
        )
        output = model(xb)
        motif_probabilities.append(
            torch.softmax(output["motif_logits"], dim=1).cpu().numpy()
        )
        window_probabilities.append(
            torch.sigmoid(output["window_logits"]).cpu().numpy()
        )
        offtarget_z.append(output["offtarget_z"].cpu().numpy())

    motif_probabilities = np.concatenate(motif_probabilities, axis=0)
    window_probabilities = np.concatenate(window_probabilities, axis=0)
    offtarget_z = np.concatenate(offtarget_z, axis=0)

    offtarget = (
        offtarget_z * float(norm["offtarget_std"])
        + float(norm["offtarget_mean"])
    )
    offtarget = np.clip(offtarget, 0.0, 1.0)

    starts = []
    ends = []
    for probabilities in window_probabilities:
        start, end = decode_contiguous_window(
            probabilities,
            window_threshold,
        )
        starts.append(start)
        ends.append(end)

    return {
        "motif_probabilities": motif_probabilities,
        "motif_index": np.argmax(motif_probabilities, axis=1),
        "window_probabilities": window_probabilities,
        "window_start": np.asarray(starts, dtype=int),
        "window_end": np.asarray(ends, dtype=int),
        "offtarget": offtarget,
    }


def build_parser():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input-txt",
        required=True,
        type=Path,
        help="Two-column TXT: first column name, second column protein sequence.",
    )
    ap.add_argument("--output-tsv", required=True, type=Path)
    ap.add_argument("--has-header", action="store_true")

    ap.add_argument(
        "--ontarget-script",
        type=Path,
        required=True,
        help="Path to predict_final_bilstm_txt.py.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to final_bilstm_manifest.json.",
    )
    ap.add_argument(
        "--esm-checkpoint",
        type=Path,
        required=True,
        help="Path to the ESM-C model checkpoint.",
    )
    ap.add_argument("--embedding-cache-pkl", type=Path, default=None)
    ap.add_argument("--regenerate-embeddings", action="store_true")
    ap.add_argument("--disable-flash-attn", action="store_true")

    ap.add_argument("--specificity-checkpoint", required=True, type=Path)
    ap.add_argument("--specificity-scaler", type=Path, default=None)
    ap.add_argument("--window-threshold", type=float, default=None)

    ap.add_argument("--esm-device", default="cuda")
    ap.add_argument("--predict-device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)

    return ap


def main():
    args = build_parser().parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    for path in (
        args.input_txt,
        args.ontarget_script,
        args.manifest,
        args.esm_checkpoint,
        args.specificity_checkpoint,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required file is missing or empty: {path}")

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)

    cache_path = (
        args.embedding_cache_pkl
        if args.embedding_cache_pkl is not None
        else args.output_tsv.with_suffix(".embeddings.pkl")
    )

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

    print(f"ESM-C device: {esm_device}", flush=True)
    print(f"Prediction device: {predict_device}", flush=True)

    ontarget = load_ontarget_module(args.ontarget_script)

    frame = ontarget.read_name_sequence(
        args.input_txt,
        args.has_header,
    )
    print(f"Read {len(frame)} sequences from {args.input_txt}", flush=True)

    cache = ontarget.update_embedding_cache(
        frame=frame,
        cache_path=cache_path,
        checkpoint=args.esm_checkpoint,
        esm_device=esm_device,
        use_flash_attn=not args.disable_flash_attn,
        regenerate=args.regenerate_embeddings,
    )

    target_dim = read_target_dim_from_manifest(args.manifest)

    features, embedding_shapes = ontarget.features_from_cache(
        frame,
        cache,
        target_dim,
    )

    seed_predictions, verified_target_dim = ontarget.predict_all_seeds(
        manifest_path=args.manifest,
        features=features,
        batch_size=args.batch_size,
        device=predict_device,
    )

    if int(verified_target_dim) != int(target_dim):
        raise RuntimeError("Inconsistent target_dim returned by on-target predictor.")

    ordered_seeds = sorted(seed_predictions)
    seed_matrix = np.column_stack(
        [seed_predictions[seed] for seed in ordered_seeds]
    )

    predicted_efficiency = seed_matrix.mean(axis=1)
    prediction_sd = (
        seed_matrix.std(axis=1, ddof=1)
        if seed_matrix.shape[1] > 1
        else np.zeros(len(frame), dtype=float)
    )
    prediction_min = seed_matrix.min(axis=1)
    prediction_max = seed_matrix.max(axis=1)

    spec_scaler_path = (
        args.specificity_scaler
        if args.specificity_scaler is not None
        else infer_specificity_scaler(args.specificity_checkpoint)
    )

    (
        specificity_model,
        specificity_scaler,
        target_normalizer,
        motif_classes,
        specificity_metadata,
        specificity_cfg,
    ) = load_specificity_model(
        args.specificity_checkpoint,
        spec_scaler_path,
        predict_device,
    )

    if int(specificity_cfg.input_dim) != int(target_dim):
        raise ValueError(
            "On-target and specificity preprocessing dimensions differ: "
            f"on-target target_dim={target_dim}, "
            f"specificity input_dim={specificity_cfg.input_dim}."
        )

    saved_threshold = float(
        specificity_metadata.get("window_threshold", 0.5)
    )
    window_threshold = (
        float(args.window_threshold)
        if args.window_threshold is not None
        else saved_threshold
    )

    if not (0.0 < window_threshold < 1.0):
        raise ValueError("Window threshold must be between 0 and 1.")

    specificity = predict_specificity(
        model=specificity_model,
        scaler=specificity_scaler,
        norm=target_normalizer,
        features=features,
        device=predict_device,
        batch_size=args.batch_size,
        window_threshold=window_threshold,
    )

    result = frame.copy()
    result["sequence_length"] = result["sequence"].str.len()
    result["raw_embedding_shape"] = embedding_shapes

    for seed in ordered_seeds:
        result[f"ontarget_prediction_seed_{seed}"] = seed_predictions[seed]

    result["AlphaCD2_ontarget"] = predicted_efficiency
    result["AlphaCD2_ontarget_percent"] = predicted_efficiency * 100.0
    result["AlphaCD2_ontarget_sd"] = prediction_sd
    result["AlphaCD2_ontarget_sd_percent"] = prediction_sd * 100.0
    result["AlphaCD2_ontarget_min"] = prediction_min
    result["AlphaCD2_ontarget_max"] = prediction_max

    result["Motif"] = [
        motif_classes[int(i)]
        for i in specificity["motif_index"]
    ]
    for i, motif in enumerate(motif_classes):
        result[f"Motif_prob_{motif}"] = specificity[
            "motif_probabilities"
        ][:, i]

    result["Window_start"] = specificity["window_start"]
    result["Window_end"] = specificity["window_end"]
    result["Window"] = [
        f"{int(start)}_{int(end)}"
        for start, end in zip(
            specificity["window_start"],
            specificity["window_end"],
        )
    ]
    for j, position in enumerate(WINDOW_POSITIONS):
        result[f"Window_prob_{position}"] = specificity[
            "window_probabilities"
        ][:, j]

    result["Offtarget"] = specificity["offtarget"]
    result["Offtarget_percent"] = specificity["offtarget"] * 100.0

    prediction_rank = (
        result["AlphaCD2_ontarget"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    result.insert(0, "prediction_rank", prediction_rank)

    ordered_columns = [
        "prediction_rank",
        "name",
        "AlphaCD2_ontarget",
        "Motif",
        "Window_start",
        "Window_end",
        "Window",
        "Offtarget",
    ]

    result = result[ordered_columns]
    result.to_csv(args.output_tsv, sep="\t", index=False)

    summary = {
        "input_txt": str(args.input_txt.resolve()),
        "ontarget_script": str(args.ontarget_script.resolve()),
        "manifest": str(args.manifest.resolve()),
        "esm_checkpoint": str(args.esm_checkpoint.resolve()),
        "embedding_cache_pkl": str(cache_path.resolve()),
        "specificity_checkpoint": str(args.specificity_checkpoint.resolve()),
        "specificity_scaler": str(spec_scaler_path.resolve()),
        "output_tsv": str(args.output_tsv.resolve()),
        "n_sequences": int(len(result)),
        "ontarget_n_seed_models": int(len(ordered_seeds)),
        "ontarget_seeds": [int(seed) for seed in ordered_seeds],
        "target_dim": int(target_dim),
        "window_threshold": float(window_threshold),
        "ontarget_mean": float(result["AlphaCD2_ontarget"].mean()),
        "ontarget_min": float(result["AlphaCD2_ontarget"].min()),
        "ontarget_max": float(result["AlphaCD2_ontarget"].max()),
    }

    summary_path = Path(str(args.output_tsv) + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved unified predictions: {args.output_tsv}", flush=True)


if __name__ == "__main__":
    main()
