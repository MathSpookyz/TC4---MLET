import yfinance as yf
import pandas as pd

def extract_prices(tickers, start_date, end_date):
    """
    Extrai preços históricos do Yahoo Finance
    Retorna DataFrame com MultiIndex (ticker, OHLC)
    """
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError("Nenhum dado retornado do Yahoo Finance")

    return df