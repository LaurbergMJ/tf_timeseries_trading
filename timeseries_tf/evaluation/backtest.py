import numpy as np 
import pandas as pd 

def simple_long_short_backtest(
        returns: pd.Series, 
        predicted_returns: np.ndarray,
        threshold: float = 0.0,
    ) -> pd.DataFrame:

    """
    Docstring for simple_long_short_backtest
    
    Simple backtest: 
    - long if predicted return > threshold
    - short if predicted return < -threshold
    - flat otherwise 
    """

    if len(returns) != len(predicted_returns):
        raise ValueError("Length of returns and predicted_returns must be the same.")
    
    positions = np.where(
        predicted_returns > threshold, 1,
        np.where(predicted_returns < -threshold, -1, 0)
    )

    pnl = positions * returns.values
    equity_curve = (1 + pnl).cumprod()

    result = pd.DataFrame(
        {
            'returns': returns.values,
            'position': positions,
            'pnl': pnl,
            'equity_curve': equity_curve,
        },
        index=returns.index
    )
    return result