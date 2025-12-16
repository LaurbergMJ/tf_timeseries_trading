from __future__ import annotations 

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow import keras 

from timeseries_tf.data.windowing import make_tf_dataset
from timeseries_tf.evaluation.metrics import mae, rmse 

@dataclass
class WalkForwardFold:
    fold_id: int 
    train_start: pd.Timestamp
    train_end: pd.Timestamp 
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int 
    n_test: int 
    metrics: Dict[str, float]

def make_yearly_walk_forward_folds(
    y_dates: pd.DatetimeIndex,
    min_train_years: int = 1,
    test_years: int = 1,
    step_years: int = 1,
    ) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, pd.Timestamp]]]:
    """
    Expanding window folds based on calendar years.

    Example: 
        Train: first 3 years, Test: next 1 year 
        Train: first 4 years, Test: next 1 year 
        ...
    
    Returns list of:
        (train_mask, test_mask, meta_dates)
    """

    if len(y_dates) == 0:
        raise ValueError("y_dates empty.")

    years = np.array(sorted(pd.unique(y_dates.year)))

    if len(years) < (min_train_years + test_years):
        raise ValueError("Not enough years in data for requested fold scheme")

    folds = []
    for i in range(min_train_years, len(years) - test_years + 1, step_years):
        train_years_set = set(years[:i])
        test_years_set = set(years[i : i + test_years])

        train_mask = np.isin(y_dates.year, list(train_years_set))
        test_mask = np.isin(y_dates.year, list(test_years_set))

        # meta ranges (actual date ranges covered)

        meta = {
            "train_start": y_dates[train_mask].min(),
            "train_end": y_dates[train_mask].max(), 
            "test_start": y_dates[test_mask].min(),
            "test_end": y_dates[test_mask].max(),
        }
        folds.append((train_mask, test_mask, meta))

    return folds

def walk_forward_forecast_regression(
    model_factory: Callable[[], keras.Model],
    X: np.ndarray, 
    y: np.ndarray, 
    y_dates: pd.DatetimeIndex,
    batch_size: int = 64,
    epochs: int = 50,
    patience: int = 10,
    min_train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    verbose: int = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Run walk-forward evaluation for a regression forecaster (e.g. predicting returns)

    model_factory must build and compile a *fresh* model (new weights) each fold 

    Returns:
        folds_df: per-fold metrics and date ranges
        summary_df: aggregate metrics (mean, std across folds)
    """

    folds = make_yearly_walk_forward_folds(
        y_dates = y_dates,
        min_train_years=min_train_years,
        test_years=test_years,
        step_years=step_years,
    )

    fold_rows: List[WalkForwardFold] = []

    for fold_id, (train_mask, test_mask, meta) in enumerate(folds, start=1):
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Build datasets (shuffle train only)
        train_ds = make_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
        test_ds = make_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

        model = model_factory()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        ]

        model.fit(
            train_ds,
            validation_data = test_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )

        y_pred = model.predict(X_test, verbose=0)
        y_true = y_test 

        # if horizon=1, squeeze for metrics
        y_pred_1d = np.squeeze(y_pred)
        y_true_1d = np.squeeze(y_true)

        m = {
            "mae": mae(y_true_1d, y_pred_1d),
            "rmse": rmse(y_true_1d, y_pred_1d),
            "directional_acc": float(np.mean(np.sign(y_pred_1d) == np.sign(y_true_1d))),
            "corr": float(np.corrcoef(y_pred_1d, y_true_1d)[0, 1]) if len(y_true_1d) >= 2 else float("nan")
        }

        fold_rows.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=meta["train_start"],
                train_end=meta["train_end"],
                test_start=meta["test_start"],
                test_end=meta["test_end"],
                n_train=len(X_train),
                n_test=len(X_test),
                metrics=m,
            )
        )

        # Build fold dataframe 
        folds_df = pd.DataFrame(
            [
                {
                    "fold": f.fold_id,
                    "train_start": f.train_start,
                    "train_end": f.train_end,
                    "test_start": f.test_start,
                    "test_end": f.test_end,
                    "n_train": f.n_train,
                    "n_test": f.n_test,
                    **f.metrics,
                }
                for f in fold_rows
            ]
        )

        # Aggregate summary 
        metric_cols = ["mae", "rmse", "directional_acc", "corr"]
        summary_df = pd.DataFrame(
            {
                "metric": metric_cols,
                "mean": [folds_df[c].mean() for c in metric_cols],
                "std": [folds_df[c].std() for c in metric_cols],
                "min": [folds_df[c].min() for c in metric_cols],
                "max": [folds_df[c].max() for c in metric_cols],
            }
        )

    return folds_df, summary_df


    

    
