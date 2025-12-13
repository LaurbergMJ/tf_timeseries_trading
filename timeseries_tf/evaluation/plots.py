import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

def plot_price_and_forecast(
        dates: pd.Index,
        prices: np.ndarray, 
        forecasts: np.ndarray, 
        start_idx: int = 0,
):
    
    actual = prices[start_idx + 1 : start_idx + 1 + len(forecasts)]
    forecast_dates = dates[start_idx + 1 : start_idx + 1 + len(forecasts)]

    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, label="Price", alpha=0.5)
    plt.plot(forecast_dates, actual, label="Actual (1-step ahead)")
    plt.plot(forecast_dates, forecasts, label="Forecast", linestyle='--')
    plt.legend()
    plt.title("Price vs 1-step forecast")
    plt.tight_layout()
    plt.show()