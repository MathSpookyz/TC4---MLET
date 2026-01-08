import logging
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from scrapper.scr.extract.yahoo_extractor import extract_prices
from scrapper.scr.transform.price_transformer import normalize_prices
from scrapper.scr.load.parquet_loader import (
    save_raw_parquet,
    save_processed_parquet,
    load_ticker_local
)


STORAGE_TYPE = os.getenv("STORAGE_TYPE", "local").lower()

if STORAGE_TYPE == "s3":
    S3_BUCKET = os.getenv("S3_BUCKET", "teste-s3-dados-tickers")
    RAW_PATH = f"s3://{S3_BUCKET}/raw/prices_raw.parquet"
    PROCESSED_PATH = f"s3://{S3_BUCKET}/processed/prices"
else:
    RAW_PATH = "scrapper/data/raw/prices_raw.parquet"
    PROCESSED_PATH = "scrapper/data/processed/prices"

REQUIRED_COLUMNS = {"date", "ticker", "close", "volume"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def validate_processed_schema(df: pd.DataFrame) -> None:
    """
    Garante que o DataFrame processado segue o contrato esperado
    por model_training.py e model_executor.py
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas obrigatórias ausentes após normalização: {missing}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError("Coluna 'date' deve ser datetime")

    if not pd.api.types.is_numeric_dtype(df["close"]):
        raise TypeError("Coluna 'close' deve ser numérica")

    if not pd.api.types.is_numeric_dtype(df["volume"]):
        raise TypeError("Coluna 'volume' deve ser numérica")



def scrappeData(
    tickers: list[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    logger.info("Início do pipeline")

    logger.info("Extraindo dados do Yahoo Finance")
    df_raw = extract_prices(tickers, start_date, end_date)

    logger.info("Salvando cache RAW localmente")
    save_raw_parquet(df_raw, RAW_PATH)

    logger.info("Normalizando dados")
    df_processed = normalize_prices(df_raw)


    df_processed.columns = df_processed.columns.str.lower()

    df_processed["date"] = pd.to_datetime(df_processed["date"])
    df_processed["ticker"] = df_processed["ticker"].astype(str)

    df_processed = df_processed.sort_values(
        by=["ticker", "date"]
    ).reset_index(drop=True)


    validate_processed_schema(df_processed)

    logger.info(f"Salvando dados tratados em Parquet ({STORAGE_TYPE})")
    
    for ticker in df_processed["ticker"].unique():
        ticker_df = df_processed[df_processed["ticker"] == ticker]
        save_processed_parquet(ticker_df, ticker, PROCESSED_PATH)

    logger.info("Fim do pipeline com sucesso")
    return df_processed



def get_or_scrappe_ticker(
    ticker: str,
    data_path: str = "scrapper/data/processed_prices",
    start_date: str | None = None,
    end_date: str | None = None
) -> pd.DataFrame:
    """
    Recupera dados processados localmente.
    Caso não existam, executa o pipeline de scraping.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date is None:
        start_date = (
            datetime.now() - timedelta(days=730)
        ).strftime("%Y-%m-%d")

    df_existing = load_ticker_local(ticker, data_path)

    if df_existing is not None:
        logger.info(f"Dados locais encontrados para {ticker}")

        df_existing.columns = df_existing.columns.str.lower()
        df_existing["date"] = pd.to_datetime(df_existing["date"])
        df_existing = df_existing.sort_values("date")

        validate_processed_schema(df_existing)
        return df_existing

    logger.info(
        f"Ticker {ticker} não encontrado localmente. "
        f"Executando extração..."
    )

    try:
        df = scrappeData([ticker], start_date, end_date)

        return df[df["ticker"] == ticker].copy()

    except Exception as e:
        logger.error(
            f"Erro ao extrair dados do ticker {ticker}: {e}"
        )
        raise ValueError(
            f"Não foi possível obter dados para o ticker {ticker}"
        ) from e
