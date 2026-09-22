#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy-compatible AlphaCD2 models for seed-42 cross-validation.

The preprocessing is intentionally kept identical to the successful legacy
pipeline: residue embeddings are flattened and truncated/padded to 1152 values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


@dataclass
class ModelConfig:
    input_dim: int = 1152
    mode: str = "ensemble"
    transformer_d_model: int = 256
    transformer_nhead: int = 8
    transformer_num_layers: int = 3
    cnn_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    bilstm_hidden_dim: int = 256
    bilstm_num_layers: int = 2
    fusion_dims: Tuple[int, ...] = (512, 256)
    dropout: float = 0.30
    max_value: float = 0.90

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["cnn_hidden_dims"] = list(self.cnn_hidden_dims)
        d["fusion_dims"] = list(self.fusion_dims)
        return d


class FlatEmbeddingDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, training: bool = False):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.training = bool(training)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].copy()
        y = float(self.y[idx])
        if self.training:
            if np.random.rand() < 0.30:
                x += np.random.normal(0.0, 0.02 * (1.0 + y), x.shape)
            if np.random.rand() < 0.20:
                mask = np.random.rand(*x.shape) < (0.05 + 0.10 * y)
                x[mask] = 0.0
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)


class HighDimTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.transformer_d_model
        self.input_projection = nn.Sequential(
            nn.Linear(cfg.input_dim, d * 2),
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * 2, d),
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.transformer_nhead,
            dim_feedforward=d * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.transformer_num_layers)
        self.feature_extractor = nn.Sequential(
            nn.Linear(d, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        x = self.input_projection(x).unsqueeze(1) + self.pos_encoding
        x = self.encoder(x).mean(dim=1)
        return self.feature_extractor(x)


class HighDimCNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # cnn_hidden_dims is retained in config for compatibility with old CLI.
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(cfg.dropout), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(cfg.dropout), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        x = self.conv(x.unsqueeze(1)).reshape(x.shape[0], -1)
        return self.fc(x)


class HighDimBiLSTM(nn.Module):
    def __init__(self, cfg: ModelConfig):
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
            nn.Linear(h * 2, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.feature_extractor = nn.Sequential(
            nn.Linear(h * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        x = self.sequence_creator(x).reshape(x.shape[0], 4, self.h)
        x, _ = self.lstm(x)
        a = torch.softmax(self.attention(x).squeeze(-1), dim=1)
        x = torch.sum(a.unsqueeze(-1) * x, dim=1)
        return self.feature_extractor(x)


class AlphaCD2Regressor(nn.Module):
    """Transformer, CNN, BiLSTM or the original three-branch ensemble."""

    MODES = {"transformer", "cnn", "bilstm", "ensemble"}

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.mode not in self.MODES:
            raise ValueError(f"Unsupported mode: {cfg.mode}")
        self.cfg = cfg
        self.mode = cfg.mode
        self.max_value = float(cfg.max_value)

        if cfg.mode in {"transformer", "ensemble"}:
            self.transformer = HighDimTransformer(cfg)
        if cfg.mode in {"cnn", "ensemble"}:
            self.cnn = HighDimCNN(cfg)
        if cfg.mode in {"bilstm", "ensemble"}:
            self.bilstm = HighDimBiLSTM(cfg)

        if cfg.mode == "ensemble":
            self.branch_logits = nn.Parameter(torch.ones(3))
            fusion_input = 128 * 3
        else:
            fusion_input = 128

        fusion = []
        for hidden in cfg.fusion_dims:
            fusion.extend([
                nn.Linear(fusion_input, hidden),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ])
            fusion_input = hidden
        self.fusion = nn.Sequential(*fusion)
        self.output = nn.Sequential(
            nn.Linear(fusion_input, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout / 2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        info = {}
        if self.mode == "transformer":
            z = self.transformer(x)
        elif self.mode == "cnn":
            z = self.cnn(x)
        elif self.mode == "bilstm":
            z = self.bilstm(x)
        else:
            zt, zc, zl = self.transformer(x), self.cnn(x), self.bilstm(x)
            w = torch.softmax(self.branch_logits, dim=0)
            z = torch.cat((zt * w[0], zc * w[1], zl * w[2]), dim=1)
            info["branch_weights"] = w
        prediction = self.output(self.fusion(z)) * self.max_value
        return prediction, info
