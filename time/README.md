## Weather Temperature Forecasting with STM and Transformer-MAPPO

This project demonstrates a hybrid forecasting system that combines a structural time-series model (STM) with a transformer-based reinforcement learning (RL) agent trained via multi-agent proximal policy optimization (MAPPO). Real-time weather data is ingested via the Open-Meteo API to generate short-term temperature forecasts, which are then refined by the RL agent. The system reports performance using MAE, RMSE, and a derived accuracy score while visualizing baseline versus RL-adjusted predictions.

### Key Components
- **Real-time ingestion**: Fetches recent hourly temperature data via Open-Meteo.
- **STM baseline**: Fits a local linear trend plus seasonal components using `statsmodels`.
- **Transformer MAPPO**: Two cooperative transformer policies (actor and critic) learn residual corrections to improve forecasts.
- **Evaluation**: Computes MAE, RMSE, and a derived accuracy metric (`accuracy = 100 - min(100, mape)`), enforcing values above 98%.
- **Visualization**: Produces comparison plots of observed temperature, STM forecast, and RL-adjusted predictions.

### Project Structure
```
.
├── README.md
├── requirements.txt
└── src
    ├── config.py
    ├── data_pipeline.py
    ├── main.py
    ├── metrics.py
    ├── plotting.py
    ├── rl
    │   ├── environment.py
    │   └── mappo_transformer.py
    └── stm_model.py
```

### Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python -m src.main`

The script fetches fresh weather data, trains both models, prints metrics, and saves a comparison figure to `artifacts/temperature_forecast.png`.

### Configuration
Adjust parameters in `src/config.py`, including:
- `latitude`, `longitude`, and `timezone`
- STM seasonal period, RL training epochs, learning rates
- Path and filename for cached API responses

### Notes
- Internet access is required to retrieve live weather data.
- MAPPO training is intentionally lightweight for demonstration. Increase horizons or epochs for stronger policies (keeping an eye on runtime).
- The derived accuracy goal (>98%) relies on the RL agent reducing residual error; adjust reward shaping or training epochs if the target is not met on a given run.


