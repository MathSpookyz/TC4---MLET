import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent / "model"))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXPORT_DIR = os.path.join(BASE_DIR, "export")
DATA_DIR = os.path.join(
    BASE_DIR,
    "scrapper",
    "data",
    "processed",
    "prices"
)

DEVICE = torch.device("cpu")

FEATURE_COLUMNS = ["close", "volume"]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


_models_cache = {}


def load_artifacts(ticker: str):
    """Carrega modelo e scalers específicos para um ticker"""
    ticker = ticker.upper()
    logger.info(f"Iniciando carregamento de artefatos para ticker: {ticker}")
    
    logger.debug(f"Verificando cache de modelos para {ticker}")
    if ticker in _models_cache:
        logger.info(f"Modelo {ticker} encontrado em cache")
        return _models_cache[ticker]
    
    logger.debug(f"Modelo {ticker} não encontrado em cache, carregando do disco")

    MODEL_PATH = os.path.join(EXPORT_DIR, f"lstm_model_{ticker}.pth")
    SCALER_FEATURES_PATH = os.path.join(EXPORT_DIR, f"scaler_features_{ticker}.save")
    SCALER_CLOSE_PATH = os.path.join(EXPORT_DIR, f"scaler_close_{ticker}.save")
    
    logger.debug(f"Verificando existência de arquivos do modelo em: {EXPORT_DIR}")

    logger.debug(f"Verificando se modelo existe: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Modelo não encontrado para {ticker}. Iniciando treinamento automático...")
        
        try:
   
            from model.model_training import train_model
            
            logger.info(f"Treinando modelo automaticamente para {ticker}")
            train_result = train_model(ticker=ticker)
            logger.info(f"Treinamento automático concluído para {ticker} com RMSE: {train_result.get('rmse', 'N/A')}")
            
            logger.debug(f"Verificando se modelo foi criado após treinamento: {MODEL_PATH}")
            if not os.path.exists(MODEL_PATH):
                error_msg = f"Falha ao criar modelo para {ticker} após treinamento automático"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
        except ImportError as e:
            error_msg = f"Erro ao importar módulo de treinamento: {e}"
            logger.error(error_msg, exc_info=True)
            raise ImportError(error_msg) from e
        except Exception as e:
            error_msg = f"Erro durante treinamento automático de {ticker}: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    logger.debug(f"Verificando se scalers existem: {SCALER_FEATURES_PATH}, {SCALER_CLOSE_PATH}")
    if not os.path.exists(SCALER_FEATURES_PATH) or not os.path.exists(SCALER_CLOSE_PATH):
        error_msg = f"Scalers não encontrados para {ticker} em {EXPORT_DIR}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Carregando checkpoint do modelo de: {MODEL_PATH}")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        logger.debug(f"Checkpoint carregado com sucesso para {ticker}")
    except Exception as e:
        error_msg = f"Erro ao carregar checkpoint para {ticker}: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e
    
    model_config = checkpoint["model_config"]
    seq_length = model_config["seq_length"]
    logger.debug(f"Configuração do modelo: {model_config}")
    
    logger.info(f"Criando modelo LSTM para {ticker}")
    try:
        model = LSTMModel(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"]
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.debug(f"Modelo carregado e definido para modo de avaliação")
    except Exception as e:
        error_msg = f"Erro ao criar/carregar modelo para {ticker}: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e
    
    logger.info(f"Carregando scalers para {ticker}")
    try:
        scaler_features = joblib.load(SCALER_FEATURES_PATH)
        scaler_close = joblib.load(SCALER_CLOSE_PATH)
        logger.debug(f"Scalers carregados com sucesso")
    except Exception as e:
        error_msg = f"Erro ao carregar scalers para {ticker}: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e
    
    artifacts = {
        "model": model,
        "scaler_features": scaler_features,
        "scaler_close": scaler_close,
        "model_config": model_config,
        "seq_length": seq_length
    }
    
    _models_cache[ticker] = artifacts
    logger.info(f"Artefatos para {ticker} armazenados em cache com sucesso")
    
    return artifacts

def load_processed_data(ticker: str) -> pd.DataFrame:
    logger.info(f"Carregando dados processados para ticker: {ticker}")
    ticker_dir = f"ticker={ticker}"
    full_path = os.path.join(DATA_DIR, ticker_dir)
    logger.debug(f"Caminho dos dados: {full_path}")

    logger.debug(f"Verificando se diretório existe: {full_path}")
    if not os.path.exists(full_path):
        error_msg = f"Dados processados não encontrados para {ticker} em {full_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Listando arquivos parquet em: {full_path}")
    files = [f for f in os.listdir(full_path) if f.endswith(".parquet")]
    logger.debug(f"Arquivos encontrados: {files}")
    
    if not files:
        error_msg = f"Nenhum arquivo parquet encontrado em {full_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Lendo arquivo parquet: {files[0]}")
    try:
        df = pd.read_parquet(os.path.join(full_path, files[0]))
        logger.debug(f"Dados carregados: {len(df)} registros, colunas: {list(df.columns)}")
    except Exception as e:
        error_msg = f"Erro ao ler arquivo parquet para {ticker}: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

    required_cols = {"date", "close", "volume"}
    logger.debug(f"Verificando colunas obrigatórias: {required_cols}")
    if not required_cols.issubset(df.columns):
        error_msg = f"Colunas esperadas ausentes para {ticker}. Esperado: {required_cols}, Encontrado: {set(df.columns)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Convertendo coluna 'date' para datetime")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Dados processados carregados com sucesso: {len(df)} registros de {df['date'].min()} até {df['date'].max()}")

    return df

def filter_date_range(df: pd.DataFrame, start: datetime, end: datetime, seq_length: int) -> pd.DataFrame:
    logger.info(f"Filtrando dados de {start} até {end}")
    logger.debug(f"Registros totais antes do filtro: {len(df)}")
    
    mask = (df["date"] >= start) & (df["date"] <= end)
    filtered = df.loc[mask]
    logger.debug(f"Registros após filtro: {len(filtered)}")

    logger.debug(f"Verificando se há dados suficientes (mínimo {seq_length} registros)")
    if len(filtered) < seq_length:
        error_msg = f"Intervalo insuficiente: encontrados {len(filtered)} registros, mínimo necessário {seq_length}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Dados filtrados com sucesso: {len(filtered)} registros")
    return filtered

def prepare_input_window(df: pd.DataFrame, scaler_features, seq_length: int) -> torch.Tensor:
    logger.debug(f"Preparando janela de entrada com {seq_length} registros")
    logger.debug(f"Colunas de features: {FEATURE_COLUMNS}")
    
    try:
        features = df[FEATURE_COLUMNS].values
        logger.debug(f"Features extraídas: shape {features.shape}")
        
        scaled = scaler_features.transform(features)
        logger.debug(f"Features escaladas: shape {scaled.shape}")

        window = scaled[-seq_length:]
        window = np.expand_dims(window, axis=0)
        logger.debug(f"Janela preparada: shape {window.shape}")

        tensor = torch.tensor(window, dtype=torch.float32, device=DEVICE)
        logger.debug(f"Tensor criado: shape {tensor.shape}, device {DEVICE}")
        return tensor
    except Exception as e:
        error_msg = f"Erro ao preparar janela de entrada: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

def predict_single_date(
    ticker: str,
    start_date: str,
    end_date: str,
    horizon: int
) -> dict:
    if horizon not in (1, 2, 7):
        raise ValueError("Horizon permitido: 1, 2 ou 7 dias")

    artifacts = load_artifacts(ticker)
    model = artifacts["model"]
    scaler_close = artifacts["scaler_close"]
    scaler_features = artifacts["scaler_features"]
    seq_length = artifacts["seq_length"]
    model_config = artifacts["model_config"]

    df = load_processed_data(ticker)

    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    df = filter_date_range(df, start, end, seq_length)

    input_tensor = prepare_input_window(df, scaler_features, seq_length)

    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy()[0][0]

    pred_price = scaler_close.inverse_transform([[pred_scaled]])[0][0]

    target_date = end + timedelta(days=horizon)

    return {
        "ticker": ticker,
        "target_date": target_date.strftime("%Y-%m-%d"),
        "horizon_days": horizon,
        "predicted_price": float(round(pred_price, 2)),
        "model_version": model_config.get("version", "v1")
    }

def predict_series(
    ticker: str,
    start_date: str,
    end_date: str,
    horizon: int
) -> dict:
    if horizon not in (1, 2, 7):
        raise ValueError("Horizon permitido: 1, 2 ou 7 dias")

    artifacts = load_artifacts(ticker)
    model = artifacts["model"]
    scaler_close = artifacts["scaler_close"]
    scaler_features = artifacts["scaler_features"]
    seq_length = artifacts["seq_length"]
    model_config = artifacts["model_config"]

    df = load_processed_data(ticker)

    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    df = filter_date_range(df, start, end, seq_length)

    predictions = []
    current_df = df.copy()

    for step in range(horizon):
        input_tensor = prepare_input_window(current_df, scaler_features, seq_length)

        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy()[0][0]

        pred_price = scaler_close.inverse_transform([[pred_scaled]])[0][0]

        next_date = current_df["date"].iloc[-1] + timedelta(days=1)

        predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "price": float(round(pred_price, 2))
        })

        new_row = current_df.iloc[-1].copy()
        new_row["date"] = next_date
        new_row["close"] = pred_price

        current_df = pd.concat(
            [current_df, pd.DataFrame([new_row])],
            ignore_index=True
        )

    return {
        "ticker": ticker,
        "horizon_days": horizon,
        "predictions": predictions,
        "model_version": model_config.get("version", "v1")
    }

def predict_price(df_processed: pd.DataFrame, ticker: str, days: int = 1) -> dict:
    """
    Função simplificada para previsão de preços usada pela API.
    
    Args:
        df_processed: DataFrame com dados processados do ticker
        ticker: Código da ação
        days: Número de dias para prever (padrão: 1)
    
    Returns:
        dict com predictions (lista de preços), days e last_known_price
    """
    ticker = ticker.upper()
    logger.info(f"Iniciando previsão de preço para {ticker} - {days} dias")
    
    logger.debug(f"Carregando artefatos do modelo para {ticker}")
    try:
        artifacts = load_artifacts(ticker)
        model = artifacts["model"]
        scaler_close = artifacts["scaler_close"]
        scaler_features = artifacts["scaler_features"]
        seq_length = artifacts["seq_length"]
        logger.debug(f"Artefatos carregados, seq_length: {seq_length}")
    except Exception as e:
        error_msg = f"Erro ao carregar artefatos para {ticker}: {e}"
        logger.error(error_msg, exc_info=True)
        raise
    
    logger.debug(f"Preparando dados para previsão")
    df = df_processed.copy()
    df.columns = df.columns.str.lower()
    
    logger.debug(f"Verificando se coluna 'date' existe: {'date' in df.columns}")
    if "date" in df.columns:
        logger.debug(f"Convertendo e ordenando por data")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    
    logger.debug(f"Verificando quantidade de dados: {len(df)} registros disponíveis, {seq_length} necessários")
    if len(df) < seq_length:
        error_msg = f"Dados insuficientes para {ticker}. Mínimo necessário: {seq_length} registros. Disponível: {len(df)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    last_known_price = float(df["close"].iloc[-1])
    logger.info(f"Último preço conhecido de {ticker}: R$ {last_known_price:.2f}")
    
    predictions = []
    current_df = df.copy()
    logger.info(f"Iniciando previsões recursivas para {days} dias")
    
    for step in range(days):
        logger.debug(f"Previsão do dia {step + 1}/{days}")
        
        try:
            logger.debug(f"Preparando janela de entrada para dia {step + 1}")
            features = current_df[FEATURE_COLUMNS].values
            scaled = scaler_features.transform(features)
            window = scaled[-seq_length:]
            window = np.expand_dims(window, axis=0)
            input_tensor = torch.tensor(window, dtype=torch.float32, device=DEVICE)
            
            logger.debug(f"Executando modelo de previsão")
            with torch.no_grad():
                pred_scaled = model(input_tensor).cpu().numpy()[0][0]
            
            pred_price = scaler_close.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(float(pred_price))
            logger.debug(f"Previsão dia {step + 1}: R$ {pred_price:.2f}")
            
            if step < days - 1:
                logger.debug(f"Adicionando previsão ao dataset para próxima iteração")
                new_row = current_df.iloc[-1].copy()
                new_row["close"] = pred_price
                
                current_df = pd.concat(
                    [current_df, pd.DataFrame([new_row])],
                    ignore_index=True
                )
        except Exception as e:
            error_msg = f"Erro ao fazer previsão do dia {step + 1} para {ticker}: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    logger.info(f"Previsões concluídas para {ticker}: {[f'R$ {p:.2f}' for p in predictions]}")
    return {
        "predictions": predictions,
        "days": days,
        "last_known_price": last_known_price
    }
