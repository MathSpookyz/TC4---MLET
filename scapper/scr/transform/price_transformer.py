import pandas as pd
import numpy as np

def tratar_prices(df_raw):
    """
    Normaliza, trata e cria features financeiras
    """
    dfs = []

    for ticker in df_raw.columns.levels[0]:
        df = df_raw[ticker].copy()

        # Seleção explícita de colunas
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

        # Index -> coluna
        df.reset_index(inplace=True)

        # Padronização de nomes
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Tipagem
        df["date"] = pd.to_datetime(df["date"])
        df["volume"] = df["volume"].astype("int64")

        # Identificação do ativo
        df["ticker"] = ticker

        # Features financeiras
        df["retorno_diario"] = df["adj_close"].pct_change()
        df["mm_20"] = df["adj_close"].rolling(20).mean()
        df["mm_50"] = df["adj_close"].rolling(50).mean()
        df["vol_20"] = df["retorno_diario"].rolling(20).std()

        # Remove linhas inválidas
        df.dropna(inplace=True)

        # Ordenação
        df.sort_values("date", inplace=True)

        dfs.append(df)

    df_final = pd.concat(dfs, ignore_index=True)

    return df_final