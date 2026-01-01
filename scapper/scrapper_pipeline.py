import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scr.extract.yahoo_extractor import extract_prices
from scr.transform.price_transformer import normalize_prices
from scr.load.parquet_loader import (
    save_raw_parquet,
    save_processed_parquet,
    load_ticker_local
)


RAW_PATH = "s3://teste-s3-dados-tickers/raw/prices_raw.parquet"
PROCESSED_PATH = "s3://teste-s3-dados-tickers/processed/prices/"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def scrappeData(tickers, start_date, end_date):
    logger.info("Início do pipeline")
    
    logger.info("Extraindo dados do Yahoo Finance")
    df_raw = extract_prices(tickers, start_date, end_date)

    logger.info("Salvando cache RAW no S3")
    save_raw_parquet(df_raw, RAW_PATH)

    logger.info("Tratando dados")
    df_processed = normalize_prices(df_raw)

    logger.info("Salvando dados tratados em Parquet no S3")
    save_processed_parquet(df_processed, PROCESSED_PATH)

    logger.info("Fim do pipeline com sucesso")
    return df_processed


def get_or_scrappe_ticker(
        ticker: str, 
        data_path: str = "scapper/data/processed/prices", 
        start_date: str = None, 
        end_date: str = None
        ) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    df_existing = load_ticker_local(ticker, data_path)
    
    if df_existing is not None:
        return df_existing
    
    logger.info(f"Ticker {ticker} não encontrado. Realizando extração de dados...")
    try:
        return scrappeData([ticker], start_date, end_date)
        
    except Exception as e:
        logger.error(f"Erro ao extrair dados do ticker {ticker}: {e}")
        raise ValueError(f"Não foi possível obter dados para o ticker {ticker}: {e}")
