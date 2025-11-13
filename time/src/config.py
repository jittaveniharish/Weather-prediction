"""
Global configuration for the temperature forecasting pipeline.
"""
from pathlib import Path


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Weather data acquisition
LATITUDE = 28.6139  # New Delhi, India
LONGITUDE = 77.2090
TIMEZONE = "Asia/Kolkata"
HOURS_BACK = 120  # last 5 days of hourly data

# Baseline (BiLSTM) hyperparameters
BASELINE_WINDOW = 24  # hours of history per sample
BASELINE_HIDDEN_SIZE = 64
BASELINE_NUM_LAYERS = 2
BASELINE_EPOCHS = 40
BASELINE_BATCH_SIZE = 32
BASELINE_LR = 1e-3

# Reinforcement learning hyperparameters
RL_CONTEXT_WINDOW = 24
RL_SEQUENCE_HORIZON = 24  # multi-step prediction horizon (hours)
RL_ACTION_SCALE = 5.0  # degrees Celsius
RL_EPOCHS = 35
RL_BATCH_SIZE = 16
RL_GAMMA = 0.99
RL_LR = 1e-4
RL_CLIP_RANGE = 0.2

# Evaluation and visualization
METRICS = ("mae", "rmse", "mape", "accuracy")
PLOT_PATH = ARTIFACTS_DIR / "temperature_forecast.png"

# Cache handling
CACHE_PATH = ARTIFACTS_DIR / "weather_cache.parquet"
CACHE_MAX_AGE_HOURS = 3


