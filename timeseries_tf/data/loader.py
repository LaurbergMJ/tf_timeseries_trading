import pandas as pd 
from pathlib import Path

def load_ohlc_csv(
        csv_path: str | Path,
        date_col: str = "date",
        columns_expected: list[str] | None = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
    
    df = pd.read_csv(csv_path, parse_dates=[date_col])

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in CSV file.")
    
    # Parse dates and sort 
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)

    if columns_expected is None:
        columns_expected = ["open", "high", "low", "close", "volume"]

    missing = [c for c in columns_expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV file: {missing}")
    
    return df 



    