import logging
from scr.extract.yahoo_extractor import extract_prices
from scr.transform.price_transformer import tratar_prices
from scr.load.parquet_loader import (
    save_raw_parquet,
    save_processed_parquet,
)

RAW_PATH = "s3://teste-s3-dados-tickers/raw/prices_raw.parquet"
PROCESSED_PATH = "s3://teste-s3-dados-tickers/processed/prices/"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Início do pipeline")

    tickers_input = input(
        "Informe o(s) ticker(s) separados por vírgula (ex: PETR4.SA,VALE3.SA): "
    )

    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    start_date = "2019-01-01"
    end_date = "2024-01-01"

    logger.info(f"Tickers selecionados: {tickers}")

    logger.info("Extraindo dados do Yahoo Finance")
    df_raw = extract_prices(tickers, start_date, end_date)

    logger.info("Salvando cache RAW no S3")
    save_raw_parquet(df_raw, RAW_PATH)

    logger.info("Tratando dados")
    df_processed = tratar_prices(df_raw)

    logger.info("Salvando dados tratados em Parquet no S3")
    save_processed_parquet(df_processed, PROCESSED_PATH)

    logger.info("Fim do pipeline com sucesso")

if __name__ == "__main__":
    main()