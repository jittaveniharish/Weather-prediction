"""
Bidirectional LSTM forecaster for temperature time series.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import config


class _SequenceDataset(Dataset):
    def __init__(self, series: np.ndarray, window: int):
        super().__init__()
        if len(series) <= window:
            raise ValueError("Series length must exceed window to build training samples.")
        self.window = window
        self.series = series.astype(np.float32)

    def __len__(self) -> int:
        return len(self.series) - self.window

    def __getitem__(self, idx: int):
        window_slice = self.series[idx : idx + self.window]
        target = self.series[idx + self.window]
        window_slice = window_slice.reshape(-1, 1)
        return window_slice, np.array([target], dtype=np.float32)


class _BILSTMModule(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        output = self.dropout(output[:, -1, :])
        return self.fc(output)


@dataclass
class _Scaler:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class BILSTMForecaster:
    """
    Trainable BiLSTM forecaster that produces one-step and multi-step predictions.
    """

    def __init__(
        self,
        window: int = config.BASELINE_WINDOW,
        hidden_size: int = config.BASELINE_HIDDEN_SIZE,
        num_layers: int = config.BASELINE_NUM_LAYERS,
        device: str | torch.device = "cpu",
    ):
        self.window = window
        self.device = torch.device(device)
        self.model = _BILSTMModule(hidden_size=hidden_size, num_layers=num_layers).to(self.device)
        self.scaler: _Scaler | None = None
        self._trained = False

    def _build_dataloader(self, series: np.ndarray) -> DataLoader:
        dataset = _SequenceDataset(series, self.window)
        return DataLoader(
            dataset,
            batch_size=config.BASELINE_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

    def fit(self, history: np.ndarray) -> None:
        history = np.asarray(history, dtype=np.float32)
        if len(history) <= self.window:
            raise ValueError("History must be longer than the window to fit the forecaster.")

        mean = float(history.mean())
        std = float(history.std() if history.std() > 1e-6 else 1.0)
        if std < 1e-6:
            std = 1.0
        self.scaler = _Scaler(mean=mean, std=std)

        normalized = self.scaler.transform(history)
        dataloader = self._build_dataloader(normalized)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.BASELINE_LR)

        self.model.train()
        for _ in range(config.BASELINE_EPOCHS):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

        self._trained = True
        self.model.eval()

    def _ensure_trained(self) -> None:
        if not self._trained or self.scaler is None:
            raise RuntimeError("BILSTMForecaster must be fitted before forecasting.")

    def _prepare_context(self, context: np.ndarray) -> np.ndarray:
        context = np.asarray(context, dtype=np.float32)
        if len(context) >= self.window:
            return context[-self.window :]
        if len(context) == 0:
            raise ValueError("Context must contain at least one value.")
        pad_value = context[0]
        padding = np.full(self.window - len(context), pad_value, dtype=np.float32)
        return np.concatenate([padding, context])

    def predict_next(self, context: np.ndarray) -> float:
        self._ensure_trained()
        prepared = self._prepare_context(context)
        normalized = self.scaler.transform(prepared)
        tensor = torch.tensor(normalized, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_norm = self.model(tensor).cpu().numpy()[0, 0]
        return float(self.scaler.inverse(np.array([pred_norm]))[0])

    def forecast(self, history: np.ndarray, steps: int) -> np.ndarray:
        self._ensure_trained()
        if steps <= 0:
            return np.empty((0,), dtype=np.float32)

        history = np.asarray(history, dtype=np.float32)
        buffer = history.copy()
        forecasts = []
        for _ in range(steps):
            context = self._prepare_context(buffer)
            next_value = self.predict_next(context)
            forecasts.append(next_value)
            buffer = np.append(buffer, next_value)
        return np.array(forecasts, dtype=np.float32)


