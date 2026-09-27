#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train AlphaCD2 BiLSTM models and save checkpoints for inference."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from alphacd2_seed42_model import (
    AlphaCD2Regressor,
    FlatEmbeddingDataset,
    ModelConfig,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_legacy_data(path: Path, target_dim: int):
    """Load embeddings and truncate or zero-pad to the input dimension."""
    with path.open("rb") as handle:
        embedding_dict = pickle.load(handle)

    ids: List[str] = []
    sequences: List[str] = []
    features: List[np.ndarray] = []
    targets: List[float] = []
    shape_counts: Dict[str, int] = {}
    skipped: List[str] = []

    for seq_id, record in embedding_dict.items():
        if (
            not isinstance(record, dict)
            or "embedding" not in record
            or "efficiency" not in record
        ):
            skipped.append(str(seq_id))
            continue

        embedding = record["embedding"]
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        embedding = np.asarray(embedding)

        shape_key = "x".join(map(str, embedding.shape))
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1

        flattened = embedding.flatten()
        if flattened.size >= target_dim:
            flattened = flattened[:target_dim]
        else:
            flattened = np.pad(
                flattened,
                (0, target_dim - flattened.size),
                mode="constant",
            )

        ids.append(str(seq_id))
        sequences.append(str(record.get("sequence", "")))
        features.append(flattened.astype(np.float32))
        targets.append(float(record["efficiency"]))

    if not ids:
        raise ValueError("No valid embedding/efficiency records were found.")

    return (
        np.asarray(ids, dtype=object),
        np.asarray(sequences, dtype=object),
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        shape_counts,
        skipped,
    )


def make_cfg(args, max_value: float) -> ModelConfig:
    return ModelConfig(
        input_dim=int(args.target_dim),
        mode="bilstm",
        transformer_d_model=256,
        transformer_nhead=8,
        transformer_num_layers=3,
        cnn_hidden_dims=(512, 256, 128),
        bilstm_hidden_dim=int(args.bilstm_hidden_dim),
        bilstm_num_layers=int(args.bilstm_num_layers),
        fusion_dims=tuple(args.fusion_dims),
        dropout=float(args.dropout),
        max_value=float(max_value),
    )


def make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    training: bool,
    num_workers: int,
):
    dataset = FlatEmbeddingDataset(
        features,
        targets,
        training=training,
    )
    # BatchNorm requires more than one sample per training batch.
    drop_last = bool(training and len(dataset) % batch_size == 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    targets = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        output, _ = model(batch_x)
        predictions.extend(output.detach().cpu().numpy().reshape(-1))
        targets.extend(batch_y.numpy().reshape(-1))

    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)

    return {
        "r2": float(r2_score(targets, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "mae": float(mean_absolute_error(targets, predictions)),
        "predictions": predictions,
        "targets": targets,
    }


def train_epoch_selection_fold(
    x_train,
    y_train,
    x_validation,
    y_validation,
    args,
    fold: int,
    device,
):
    fold_seed = args.seed + fold * 100
    set_seed(fold_seed)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_validation_scaled = scaler.transform(x_validation).astype(np.float32)

    # Set the output range from training labels.
    max_value = float(y_train.max() * args.max_value_multiplier)
    cfg = make_cfg(args, max_value)
    model = AlphaCD2Regressor(cfg).to(device)

    train_loader = make_loader(
        x_train_scaled,
        y_train,
        args.batch_size,
        True,
        args.num_workers,
    )
    validation_loader = make_loader(
        x_validation_scaled,
        y_validation,
        args.eval_batch_size,
        False,
        args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.cv_epochs,
        steps_per_epoch=len(train_loader),
    )
    criterion = nn.MSELoss()

    best_r2 = -float("inf")
    best_epoch = 1
    patience_counter = 0
    history = []

    for epoch in range(1, args.cv_epochs + 1):
        model.train()
        train_losses = []
        observed = []
        predicted = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output, _ = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(float(loss.detach().cpu()))
            observed.extend(batch_y.detach().cpu().numpy().reshape(-1))
            predicted.extend(output.detach().cpu().numpy().reshape(-1))

        validation = evaluate(model, validation_loader, device)
        train_r2 = float(r2_score(observed, predicted))

        history.append(
            {
                "fold": int(fold),
                "epoch": int(epoch),
                "train_loss": float(np.mean(train_losses)),
                "train_r2": train_r2,
                "validation_r2": validation["r2"],
                "validation_rmse": validation["rmse"],
                "validation_mae": validation["mae"],
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )

        if validation["r2"] > best_r2:
            best_r2 = validation["r2"]
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"  fold={fold} epoch={epoch:03d} "
                f"loss={np.mean(train_losses):.6f} "
                f"train_R2={train_r2:.4f} "
                f"val_R2={validation['r2']:.4f} "
                f"best_R2={best_r2:.4f}@{best_epoch}",
                flush=True,
            )

        if patience_counter >= args.patience:
            print(
                f"  fold={fold} early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}",
                flush=True,
            )
            break

    return {
        "fold": int(fold),
        "best_epoch": int(best_epoch),
        "best_r2": float(best_r2),
        "history": history,
    }


def select_epochs(features, targets, args, device, out_dir: Path) -> int:
    splitter = KFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.seed,
    )
    results = []

    print(
        f"Selecting BiLSTM training duration with "
        f"{args.cv_folds}-fold validation",
        flush=True,
    )

    fold_assignment = np.zeros(len(targets), dtype=int)

    for fold, (train_index, validation_index) in enumerate(
        splitter.split(features),
        start=1,
    ):
        fold_assignment[validation_index] = fold

        print(
            f"Epoch-selection fold {fold}: "
            f"train={len(train_index)}, validation={len(validation_index)}",
            flush=True,
        )

        result = train_epoch_selection_fold(
            features[train_index],
            targets[train_index],
            features[validation_index],
            targets[validation_index],
            args,
            fold,
            device,
        )
        results.append(result)

        pd.DataFrame(result["history"]).to_csv(
            out_dir / f"epoch_selection_fold{fold}.tsv",
            sep="\t",
            index=False,
        )

    pd.DataFrame(
        {
            "global_index": np.arange(len(targets)),
            "fold": fold_assignment,
            "target": targets,
        }
    ).to_csv(
        out_dir / "epoch_selection_fold_assignments.tsv",
        sep="\t",
        index=False,
    )

    best_epochs = [result["best_epoch"] for result in results]
    selected_epochs = max(1, int(round(float(np.median(best_epochs)))))

    summary = {
        "model": "bilstm",
        "fold_best_epochs": best_epochs,
        "fold_best_r2": [result["best_r2"] for result in results],
        "selected_final_epochs": selected_epochs,
        "selection_rule": "median fold-specific best epoch",
        "fold_seed": int(args.seed),
        "note": (
            "This stage selects training duration only. Formal performance "
            "must come from the previously completed nested/Group65/Strict60 "
            "OOF analyses."
        ),
    }

    (out_dir / "epoch_selection.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return selected_epochs


def train_full_seed(
    ids,
    sequences,
    features,
    targets,
    args,
    epochs: int,
    seed: int,
    device,
    out_dir: Path,
):
    set_seed(seed)
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features).astype(np.float32)

    max_value = float(targets.max() * args.max_value_multiplier)
    cfg = make_cfg(args, max_value)
    model = AlphaCD2Regressor(cfg).to(device)

    loader = make_loader(
        scaled_features,
        targets,
        args.batch_size,
        True,
        args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=epochs,
        steps_per_epoch=len(loader),
    )
    criterion = nn.MSELoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        observed = []
        predictions = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output, _ = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(float(loss.detach().cpu()))
            observed.extend(batch_y.detach().cpu().numpy().reshape(-1))
            predictions.extend(output.detach().cpu().numpy().reshape(-1))

        train_r2 = float(r2_score(observed, predictions))
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)),
                "train_r2": train_r2,
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )

        print(
            f"  final BiLSTM seed={seed} epoch={epoch:03d}/{epochs} "
            f"loss={np.mean(losses):.6f} train_R2={train_r2:.4f}",
            flush=True,
        )

    checkpoint_path = seed_dir / "bilstm_checkpoint.pt"
    scaler_path = seed_dir / "scaler.pkl"

    checkpoint = {
        "checkpoint_version": 3,
        "model_family": "AlphaCD2 legacy BiLSTM",
        "model_mode": "bilstm",
        "model_state_dict": model.state_dict(),
        "model_config": cfg.to_dict(),
        "preprocessing": {
            "method": "flatten_truncate_or_pad",
            "target_dim": int(args.target_dim),
            "legacy_compatible": True,
        },
        "training": {
            "seed": int(seed),
            "epochs": int(epochs),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "n_records": int(len(targets)),
            "target_min": float(targets.min()),
            "target_max": float(targets.max()),
            "target_mean": float(targets.mean()),
            "target_median": float(np.median(targets)),
            "max_value_multiplier": float(args.max_value_multiplier),
            "max_value": float(max_value),
        },
    }

    torch.save(checkpoint, checkpoint_path)
    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle)

    pd.DataFrame(history).to_csv(
        seed_dir / "training_history.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        {
            "id": ids,
            "sequence": sequences,
            "target": targets,
        }
    ).to_csv(
        seed_dir / "training_records.tsv",
        sep="\t",
        index=False,
    )

    return {
        "seed": int(seed),
        "checkpoint": str(checkpoint_path.relative_to(out_dir)),
        "scaler": str(scaler_path.relative_to(out_dir)),
        "epochs": int(epochs),
        "max_value": float(max_value),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-pickle", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-dim", type=int, default=1152)

    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-epochs", type=int, default=300)
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=0,
        help=(
            "Positive value skips epoch-selection CV. "
            "Zero selects the median best epoch across folds."
        ),
    )
    parser.add_argument(
        "--final-seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-value-multiplier", type=float, default=1.20)

    parser.add_argument(
        "--fusion-dims",
        type=int,
        nargs="+",
        default=[512, 256],
    )
    parser.add_argument("--bilstm-hidden-dim", type=int, default=256)
    parser.add_argument("--bilstm-num-layers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}", flush=True)

    (
        ids,
        sequences,
        features,
        targets,
        shape_counts,
        skipped,
    ) = load_legacy_data(
        args.data_pickle,
        args.target_dim,
    )

    dataset_summary = {
        "model": "bilstm",
        "n": int(len(ids)),
        "feature_shape": list(features.shape),
        "target_min": float(targets.min()),
        "target_max": float(targets.max()),
        "target_mean": float(targets.mean()),
        "target_median": float(np.median(targets)),
        "training_max_value": float(
            targets.max() * args.max_value_multiplier
        ),
        "embedding_shape_counts": shape_counts,
        "skipped_records": skipped,
    }

    (args.out_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(dataset_summary, indent=2), flush=True)

    if args.final_epochs > 0:
        selected_epochs = int(args.final_epochs)
        epoch_source = "user supplied --final-epochs"
    else:
        selected_epochs = select_epochs(
            features,
            targets,
            args,
            device,
            args.out_dir,
        )
        epoch_source = "median best epoch from random five-fold validation"

    models = []
    for seed in args.final_seeds:
        models.append(
            train_full_seed(
                ids,
                sequences,
                features,
                targets,
                args,
                selected_epochs,
                seed,
                device,
                args.out_dir,
            )
        )

    manifest = {
        "manifest_version": 3,
        "model_family": "AlphaCD2 legacy BiLSTM activity predictor",
        "model_mode": "bilstm",
        "data_pickle": str(args.data_pickle.resolve()),
        "n_training_records": int(len(ids)),
        "selected_final_epochs": int(selected_epochs),
        "epoch_selection_source": epoch_source,
        "models": models,
        "prediction_rule": (
            "Mean of all full-data seed predictions; uncertainty is the "
            "between-seed standard deviation."
        ),
        "preprocessing": "legacy embedding.flatten()[:1152] compatible",
    }

    manifest_path = args.out_dir / "final_bilstm_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
