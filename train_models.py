"""
train_models.py
---------------
Orchestrates the full pipeline:
  1. Fetch & preprocess data
  2. Train all 4 models (GARCH-LSTM, GRU, RNN, BiRNN)
  3. Evaluate on test set
  4. Generate multi-horizon forecasts (3 / 5 / 7 days)
  5. Build charts
  6. Return a single results dict consumed by Flask templates
"""

import os
import logging
import numpy as np
import tensorflow as tf

from utils.data_pipeline import fetch_and_prepare
from utils.metrics       import compute_metrics
from utils.charts        import (
    prediction_line_chart,
    model_comparison_bar_chart,
    future_forecast_chart,
)
from models.model_definitions import MODEL_REGISTRY

logger = logging.getLogger(__name__)

# ── Training hyper-parameters ─────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 64
PATIENCE   = 7


def run_pipeline(ticker: str) -> dict:
    # ── 1. Data ──────────────────────────────────────────────────────────────
    data = fetch_and_prepare(ticker)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]
    seq_len    = data["seq_len"]
    n_features = data["n_features"]

    logger.info(f"[TRAIN] X_train={X_train.shape}  X_test={X_test.shape}")

    # ── 2. Train all models ───────────────────────────────────────────────────
    predictions = {}
    metrics     = {}
    histories   = {}

    for model_name, builder_fn in MODEL_REGISTRY.items():
        logger.info(f"\n{'─'*50}")
        logger.info(f"  Training model: {model_name}")
        logger.info(f"{'─'*50}")

        model = builder_fn(seq_len, n_features)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3,
                min_lr=1e-6, verbose=1,
            ),
        ]

        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1,
        )

        # Save model
        model_dir  = os.path.join("models", "saved")
        os.makedirs(model_dir, exist_ok=True)
        save_path  = os.path.join(model_dir, f"{ticker}_{model_name}.keras")
        model.save(save_path)
        logger.info(f"[SAVE] Model saved → {save_path}")

        # Predict
        y_pred = model.predict(X_test, verbose=0).flatten()
        predictions[model_name] = y_pred
        metrics[model_name]     = compute_metrics(y_test, y_pred)
        histories[model_name]   = history.history

        logger.info(
            f"[EVAL] {model_name} → "
            f"RMSE={metrics[model_name]['rmse']:.5f}  "
            f"MAE={metrics[model_name]['mae']:.5f}  "
            f"Accuracy={metrics[model_name]['accuracy']:.1f}%"
        )

    # ── 3. Multi-horizon forecasts ────────────────────────────────────────────
    horizons = [3, 5, 7]
    horizon_labels  = [f"Day {h}" for h in horizons]
    model_forecasts = {}

    for model_name, builder_fn in MODEL_REGISTRY.items():
        save_path = os.path.join("models", "saved", f"{ticker}_{model_name}.keras")

        # ✅ FIX APPLIED HERE
        model = tf.keras.models.load_model(
            save_path,
            safe_mode=False
        )

        forecasts = _forecast_multi_horizon(model, X_test, horizons)
        model_forecasts[model_name] = forecasts

        logger.info(f"[FORECAST] {model_name}: {dict(zip(horizon_labels, forecasts))}")

    # ── 4. Charts ─────────────────────────────────────────────────────────────
    chart_predictions   = prediction_line_chart(None, y_test, predictions)
    chart_comparison    = model_comparison_bar_chart(metrics)
    chart_forecast      = future_forecast_chart(model_forecasts, horizon_labels)

    # ── 5. Consensus ──────────────────────────────────────────────────────────
    consensus_metrics = {
        "rmse":     round(np.mean([m["rmse"]     for m in metrics.values()]), 6),
        "mae":      round(np.mean([m["mae"]      for m in metrics.values()]), 6),
        "accuracy": round(np.mean([m["accuracy"] for m in metrics.values()]), 2),
    }

    best_model = max(metrics, key=lambda k: metrics[k]["accuracy"])

    # ── 6. Pack results ───────────────────────────────────────────────────────
    return {
        "ticker":           ticker,
        "metrics":          metrics,
        "consensus":        consensus_metrics,
        "best_model":       best_model,
        "forecasts":        {
            name: dict(zip(horizon_labels, vals))
            for name, vals in model_forecasts.items()
        },
        "horizon_labels":   horizon_labels,
        "chart_predictions": chart_predictions,
        "chart_comparison":  chart_comparison,
        "chart_forecast":    chart_forecast,
        "data_points": {
            "train": len(X_train),
            "test":  len(X_test),
        },
    }


def _forecast_multi_horizon(model, X_test: np.ndarray, horizons: list) -> list:
    window = X_test[-1].copy()
    preds  = []

    max_horizon = max(horizons)

    for step in range(1, max_horizon + 1):
        x_input   = window[np.newaxis, ...]
        next_pred = float(model.predict(x_input, verbose=0)[0, 0])

        new_row    = window[-1].copy()
        new_row[0] = next_pred
        window     = np.vstack([window[1:], new_row])

        if step in horizons:
            preds.append(round(next_pred, 6))

    return preds