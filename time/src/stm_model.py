"""
Compatibility shim exposing the BiLSTM forecaster under the legacy STM name.
"""
from __future__ import annotations

from .bilstm_model import BILSTMForecaster

STMForecaster = BILSTMForecaster

__all__ = ["STMForecaster"]


