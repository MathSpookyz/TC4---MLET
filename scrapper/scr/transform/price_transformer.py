import pandas as pd
import numpy as np

REQUIRED_COLUMNS = {
    "date", "ticker", "close", "volume"
}

def normalize_prices(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza dados brutos do Yahoo Finance e cria features financeiras.
    Contrato garantido para treino e inferência.
    """
    dfs = []

    for ticker in df_raw.columns.levels[0]:
        df = df_raw[ticker].copy()

        df = df[[
            "Open", "High", "Low",
            "Close", "Adj Close", "Volume"
        ]]

        df.reset_index(inplace=True)

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        df["date"] = pd.to_datetime(df["date"])
        df["volume"] = df["volume"].astype("float64")

        df["ticker"] = str(ticker)


        df["close"] = df["close"].astype("float64")

        df["retorno_diario"] = df["close"].pct_change()
        df["mm_20"] = df["close"].rolling(20).mean()
        df["mm_50"] = df["close"].rolling(50).mean()
        df["vol_20"] = df["retorno_diario"].rolling(20).std()

        df.dropna(inplace=True)

        df.sort_values("date", inplace=True)

        dfs.append(df)

    df_final = pd.concat(dfs, ignore_index=True)


    missing = REQUIRED_COLUMNS - set(df_final.columns)
    if missing:
        raise ValueError(
            f"Colunas obrigatórias ausentes após normalização: {missing}"
        )

    return df_final
