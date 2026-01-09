import pandas as pd
import os
import io
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

STORAGE_TYPE = os.getenv("STORAGE_TYPE", "local").lower()

if STORAGE_TYPE == "s3":
    try:
        import boto3
        s3_client = boto3.client("s3")
    except ImportError:
        logger.error("boto3 não está instalado. Instale com: pip install boto3")
        raise
else:
    s3_client = None



def _parse_s3_path(s3_path: str):
    """Parse s3://bucket/key em bucket e key"""
    s3_path = s3_path.replace("s3://", "")
    bucket, key = s3_path.split("/", 1)
    return bucket, key

def save_raw_parquet(df: pd.DataFrame, path: str):
    """
    Salva dados brutos em formato parquet.
    Usa STORAGE_TYPE para decidir entre S3 ou local.
    
    Args:
        df: DataFrame a ser salvo
        path: Caminho S3 (s3://bucket/key) ou caminho local
    """
    if STORAGE_TYPE == "s3":
        logger.info(f"Salvando dados brutos no S3: {path}")
        bucket, key = _parse_s3_path(path)
        
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=True)
        buffer.seek(0)
        
        s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        logger.info(f"Dados brutos salvos no S3: {path}")
    else:
        logger.info(f"Salvando dados brutos localmente: {path}")
        local_path = Path(path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(local_path, index=True)
        logger.info(f"Dados brutos salvos localmente: {local_path}")

def save_processed_parquet(df: pd.DataFrame, ticker: str, base_path: str = "scrapper/data/processed/prices"):
    """
    Salva dados processados por ticker em formato parquet.
    Usa STORAGE_TYPE para decidir entre S3 ou local.
    
    Args:
        df: DataFrame processado
        ticker: Código do ticker
        base_path: Caminho base S3 (s3://bucket/prefix/) ou caminho local
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if STORAGE_TYPE == "s3":
        if not base_path.startswith("s3://"):
            base_path = f"s3://{base_path}"
        
        bucket, prefix = _parse_s3_path(base_path)
        s3_key = f"{prefix}/ticker={ticker}/prices_processed_{timestamp}.parquet"
        
        logger.info(f"Salvando dados processados no S3: s3://{bucket}/{s3_key}")
        
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=True)
        buffer.seek(0)
        
        s3_client.put_object(Bucket=bucket, Key=s3_key, Body=buffer.getvalue())
        logger.info(f"Dados processados salvos no S3")
    else:
        ticker_path = Path(base_path) / f"ticker={ticker}"
        ticker_path.mkdir(parents=True, exist_ok=True)
        
        file_path = ticker_path / f"prices_processed_{timestamp}.parquet"
        
        logger.info(f"Salvando dados processados localmente: {file_path}")
        df.to_parquet(file_path, index=True)
        logger.info(f"Dados processados salvos localmente")

def load_ticker_local(ticker: str, data_path: str = "scrapper/data/processed/prices") -> pd.DataFrame:
    """
    Carrega dados processados de um ticker.
    Tenta primeiro localmente, depois S3 (se configurado).
    
    Args:
        ticker: Código do ticker
        data_path: Caminho base para buscar dados
    
    Returns:
        DataFrame com dados do ticker ou None se não encontrado
    """
    ticker_path = Path(data_path) / f"ticker={ticker}"
    
    if ticker_path.exists():
        parquet_files = list(ticker_path.glob("*.parquet"))
        
        if parquet_files:
            try:
                latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Ticker {ticker} encontrado localmente. Carregando: {latest_file}")
                df = pd.read_parquet(latest_file)
                
                if 'ticker' not in df.columns:
                    df['ticker'] = ticker
                    logger.info(f"Coluna 'ticker' adicionada aos dados de {ticker}")
                
                logger.info(f"Dados locais encontrados para {ticker}")
                return df
            except Exception as e:
                logger.warning(f"Erro ao carregar dados locais do ticker {ticker}: {e}")
    
    if STORAGE_TYPE == "s3":
        logger.info(f"Ticker {ticker} não encontrado localmente. Tentando carregar do S3...")
        
        try:
            if data_path.startswith("s3://"):
                bucket, prefix = _parse_s3_path(data_path)
            else:
                default_bucket = os.getenv("S3_BUCKET", "teste-s3-dados-tickers")
                bucket = default_bucket
                prefix = f"processed/prices"
            
            s3_prefix = f"{prefix}/ticker={ticker}/"
            
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
            
            if 'Contents' not in response or len(response['Contents']) == 0:
                logger.info(f"Nenhum arquivo encontrado no S3 para ticker {ticker}")
                return None
            
            parquet_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.parquet')]
            
            if not parquet_files:
                logger.info(f"Nenhum arquivo parquet encontrado no S3 para ticker {ticker}")
                return None
            
            s3_key = parquet_files[0]
            logger.info(f"Carregando ticker {ticker} do S3: s3://{bucket}/{s3_key}")
            
            obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
            df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
            
            if 'ticker' not in df.columns:
                df['ticker'] = ticker
                logger.info(f"Coluna 'ticker' adicionada aos dados de {ticker} do S3")
            
            logger.info(f"Ticker {ticker} carregado com sucesso do S3")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados do S3 para ticker {ticker}: {e}")
            return None
    else:
        logger.info(f"Ticker {ticker} não encontrado localmente em {ticker_path}")
        return None
