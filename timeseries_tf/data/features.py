import pandas as pd
import numpy as np

def add_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame.")

    df = df.copy()
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    df = df.dropna()

    return df