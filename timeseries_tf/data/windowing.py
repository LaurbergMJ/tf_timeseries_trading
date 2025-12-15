import numpy as np
import pandas as pd 
import tensorflow as tf 
from typing import Tuple

def make_supervised_windows(
        series: np.ndarray,
        window_size: int,
        horizon: int = 1, 
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create (X,y) arrays from a 1D series
    
    X shape: (num_samples, window_size, 1)
    y shape: (num_samples, horizon)
    """

    X, y = [], []
    series = np.asarray(series)

    if series.ndim != 1:
        raise ValueError("Input series must be 1-dimensional.")
    
    num_samples = len(series) - window_size - horizon + 1
    if num_samples <= 0:
        raise ValueError("Not enough data points to create a single window.")
    
    for i in range(num_samples):
        window = series[i : i + window_size]
        target = series[i + window_size : i + window_size + horizon]
        X.append(window)
        y.append(target)
    
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y


def make_tf_dataset(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(X)))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# --- Adding a windowing function for walk-forward evaluation (expanding window) ---
def make_supervised_windows_with_dates(
        series: np.ndarray, 
        dates: pd.Index, 
        window_size: int,
        horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:

    """
    Docstring for make_supervised_windows_with_dates

    Create (X, y, y_dates) from a 1D series and a matching DatetimeIndex 

    X shape: (num_samples, window_size, 1)
    y shape: (num_samples, horizon)
    y_dates: DatetimeIndex aligned to y (date of the last element of each target horizon)
    """

    series = np.asarray(series)

    if series.ndim != 1:
        raise ValueError("Input series must be 1-dimensional.")
    if len(series) != len(dates):
        raise ValueError("Length of series and dates must match.")
    
    X, y, y_dates = [], [], []

    num_samples = len(series) - window_size - horizon + 1

    if num_samples <= 0:
        raise ValueError("Not enough data for given window_size and horizon")
    
    for i in range(num_samples):
        window = series[i : i + window_size]
        target = series[i + window_size : i + window_size + horizon]

        # date of the final target element (end of horizon)
        target_end_idx = i + window_size + horizon - 1
        X.append(window)
        y.append(target)
        y_dates.append(dates[target_end_idx])

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    y_dates = pd.DatetimeIndex(y_dates)

    return X, y, y_dates
