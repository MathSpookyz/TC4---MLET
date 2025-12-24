from pathlib import Path
import pandas as pd

def save_raw_parquet(df, path):
    """
    Salva camada RAW (sem tratamento)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")

def save_processed_parquet(df, base_path):
    """
    Salva camada PROCESSED particionada por ticker
    """
    Path(base_path).mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        base_path,
        engine="pyarrow",
        partition_cols=["ticker"],
        index=False
    )

def parquet_exists(path):
    return Path(path).exists()