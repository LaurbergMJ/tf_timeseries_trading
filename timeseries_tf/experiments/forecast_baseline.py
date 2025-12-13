import json 
from pathlib import Path

import numpy as np 
import tensorflow as tf 

from timeseries_tf.data.loader import load_ohlc_csv
from timeseries_tf.data.features import add_log_returns
from timeseries_tf.data.windowing import make_supervised_windows, make_tf_dataset
from timeseries_tf.models import build_mlp_forecaster
from timeseries_tf.training.trainer import TrainingConfig, train_model
from timeseries_tf.evaluation.metrics import mae, rmse 
from timeseries_tf.evaluation.plots import plot_price_and_forecast 

def run_forecast_experiment(config_path: str = "timeseries_tf/config/example_forecast.json"):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    # --- Load data ---
    df = load_ohlc_csv(
        csv_path=data_cfg["csv_path"],
        date_column=data_cfg.get("date_column", "date"),
    )

    df = add_log_returns(df, price_col=data_cfg.get("price_column", "close"))

    # series we forecast 
    series = df["log_return"].values.astype(np.float32)

    # --- Windowing ---
    window_size = feat_cfg["window_size"]
    horizon = feat_cfg.get("horizon", 1)

    X, y = make_supervised_windows(series, window_size=window_size, horizon=horizon)

    # simple train/test split in time 
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    train_ds = make_tf_dataset(X_train, y_train, batch_size=train_cfg["batch"])
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
        train_cfg["batch_size"]
    )

    # --- Buld model ---
    if model_cfg["type"] == "mlp":
        model = build_mlp_forecaster(
            window_size=window_size,
            horizon=horizon,
            hidden_units=model_cfg.get("hidden_units", [64, 32]),
            learning_rate=train_cfg.get("learning_rate", 1e-3),
        )
    else:
        raise ValueError(f"Unknown model type: {model_cfg['type']}")
    
    # --- Train --- 
    config = TrainingConfig(
        epochs=train_cfg.["epochs"],
        batch_size=train_cfg["batch_size="],
        validation_split=train_cfg.get("validation_split", 0.2),
        patience=train_cfg.get("patience", 10),
    )

    history = train_model(
        model,
        train_ds,
        test_ds,
        config,
        use_tf_dataset=True,
    )

    # --- Evaluate ---
    y_pred = model.predict(X_test).squeeze()
    y_true = y_test.squeeze()

    print("Test MAE:", mae(y_true, y_pred))
    print("Test RMSE:", rmse(y_true, y_pred))

    # --- Plot ---
    plot_price_and_forecast(
        dates=df.index,
        prices=df["close"].values,
        forecasts=y_pred,
        start_idx=split_idx,
    )
    