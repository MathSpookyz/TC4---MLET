import boto3
import io
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

s3 = boto3.client("s3")

def save_raw_parquet(df: pd.DataFrame, s3_path: str):
    bucket, key = _parse_s3_path(s3_path)

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=True)
    buffer.seek(0)

    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def save_processed_parquet(df: pd.DataFrame, s3_prefix: str):
    bucket, prefix = _parse_s3_path(s3_prefix)

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=True)
    buffer.seek(0)

    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}prices_processed.parquet",
        Body=buffer.getvalue()
    )


def _parse_s3_path(s3_path: str):
    s3_path = s3_path.replace("s3://", "")
    bucket, key = s3_path.split("/", 1)
    return bucket, key


def load_ticker_local(
    ticker: str,
    data_path: str = "scapper/data/processed/prices", 
    s3_path: str = "s3://fiap-tech-challenge-4/processed/prices/"
) -> pd.DataFrame:

    ticker_path = Path(data_path) / f"ticker={ticker}"
    
    if ticker_path.exists():
        parquet_files = list(ticker_path.glob("*.parquet"))
        
        if parquet_files:
            try:
                logger.info(f"Ticker {ticker} encontrado localmente. Carregando dados existentes...")
                df = pd.read_parquet(parquet_files[0])
                return df
            except Exception as e:
                logger.warning(f"Erro ao carregar dados locais do ticker {ticker}: {e}")
    
    logger.info(f"Ticker {ticker} não encontrado localmente. Tentando carregar do S3...")
    
    try:
        bucket, prefix = _parse_s3_path(s3_path)
        s3_prefix = f"{prefix}ticker={ticker}/"
        
        response = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
        
        if 'Contents' not in response or len(response['Contents']) == 0:
            logger.info(f"Nenhum arquivo encontrado no S3 para ticker {ticker}")
            return None
        
        parquet_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.parquet')]
        
        if not parquet_files:
            logger.info(f"Nenhum arquivo parquet encontrado no S3 para ticker {ticker}")
            return None
        
        s3_key = parquet_files[0]
        logger.info(f"Carregando ticker {ticker} do S3: s3://{bucket}/{s3_key}")
        
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
        
        logger.info(f"Ticker {ticker} carregado com sucesso do S3")
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados do S3 para ticker {ticker}: {e}")
        return None
