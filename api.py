from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "scrapper"))
sys.path.append(str(Path(__file__).parent / "model"))

from model.model_executor import predict_price
from scrapper_pipeline import get_or_scrappe_ticker
from model_training import train_model

app = FastAPI(
    title="Stock Price Prediction API",
    description="API para previsão de preços de ações usando LSTM",
    version="1.0.0"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Configurar MLFlow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "stock-price-prediction")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
logger.info(f"MLFlow configurado - URI: {MLFLOW_TRACKING_URI}, Experimento: {MLFLOW_EXPERIMENT_NAME}")


class PredictionRequest(BaseModel):
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    days: Optional[int] = 1


class TrainRequest(BaseModel):
    ticker: str
    start_date: Optional[str] = "2020-01-01"
    end_date: Optional[str] = None


class PredictionResponse(BaseModel):
    ticker: str
    predictions: list[float]
    days: int
    last_known_price: float
    currency: str = "BRL"


class TrainResponse(BaseModel):
    ticker: str
    status: str
    message: str
    rmse: Optional[float] = None
    next_prediction: Optional[float] = None
    trained_at: Optional[str] = None


class HistoricalDataPoint(BaseModel):
    date: str
    close: float
    volume: float


class CustomPredictionRequest(BaseModel):
    historical_data: list[HistoricalDataPoint]
    days: int = 1
    ticker_name: Optional[str] = "CUSTOM"


class CustomPredictionResponse(BaseModel):
    ticker_name: str
    predictions: list[float]
    days: int
    last_known_price: float
    rmse: float
    training_samples: int
    message: str


@app.get("/")
def root():
    return {
        "message": "Stock Price Prediction API",
        "endpoints": {
            "predict": "/predict/{ticker}",
            "predict_post": "/predict",
            "predict_custom": "/predict-custom",
            "train": "/train",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/predict/{ticker}", response_model=PredictionResponse)
def predict_ticker_get(ticker: str, days: int = 1, start_date: Optional[str] = None, end_date: Optional[str] = None):
    try:
        logger.info(f"Recebida requisição de previsão para ticker: {ticker}, dias: {days}")
        
        df_processed = get_or_scrappe_ticker(
            ticker=ticker.upper(),
            data_path="scapper/data/processed/prices",
            start_date=start_date,
            end_date=end_date
        )
        
        if df_processed is None or df_processed.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Não foi possível obter dados para o ticker {ticker}"
            )
        
        result = predict_price(df_processed, ticker=ticker.upper(), days=days)
        
        logger.info(f"Previsão concluída para {ticker}: {days} dias")
        
        return PredictionResponse(
            ticker=ticker.upper(),
            predictions=[round(p, 2) for p in result['predictions']],
            days=result['days'],
            last_known_price=round(result['last_known_price'], 2)
        )
        
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao processar previsão: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
def predict_ticker_post(request: PredictionRequest):
    try:
        logger.info(f"Recebida requisição POST de previsão para ticker: {request.ticker}, dias: {request.days}")
        
        # Iniciar tracking MLFlow
        with mlflow.start_run(run_name=f"predict_POST_{request.ticker.upper()}_{request.days}days"):
            mlflow.log_param("ticker", request.ticker.upper())
            mlflow.log_param("days", request.days)
            mlflow.log_param("endpoint", "POST /predict")
            if request.start_date:
                mlflow.log_param("start_date", request.start_date)
            if request.end_date:
                mlflow.log_param("end_date", request.end_date)
            
            df_processed = get_or_scrappe_ticker(
                ticker=request.ticker.upper(),
                data_path="scapper/data/processed/prices",
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            if df_processed is None or df_processed.empty:
                mlflow.log_param("status", "error_no_data")
                raise HTTPException(
                    status_code=404,
                    detail=f"Não foi possível obter dados para o ticker {request.ticker}"
                )
            
            mlflow.log_metric("data_points", len(df_processed))
            
            result = predict_price(df_processed, ticker=request.ticker.upper(), days=request.days)
            
            # Log das previsões
            mlflow.log_metric("last_known_price", result['last_known_price'])
            for i, pred in enumerate(result['predictions'], 1):
                mlflow.log_metric(f"prediction_day_{i}", pred)
            mlflow.log_param("status", "success")
            
            logger.info(f"Previsão concluída para {request.ticker}: {request.days} dias")
            
            return PredictionResponse(
                ticker=request.ticker.upper(),
                predictions=[round(p, 2) for p in result['predictions']],
                days=result['days'],
                last_known_price=round(result['last_known_price'], 2)
            )
        
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao processar previsão: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.post("/train", response_model=TrainResponse)
def train_ticker(request: TrainRequest):
    """
    Treina um modelo LSTM para um ticker específico.
    
    O treinamento pode levar alguns minutos dependendo da quantidade de dados.
    """
    try:
        logger.info(f"Iniciando treinamento para ticker: {request.ticker}")
        
        result = train_model(
            ticker=request.ticker.upper(),
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        logger.info(f"Treinamento concluído para {request.ticker}")
        
        return TrainResponse(
            ticker=request.ticker.upper(),
            status="success",
            message=f"Modelo treinado com sucesso para {request.ticker.upper()}",
            rmse=result.get("rmse"),
            next_prediction=result.get("next_prediction"),
            trained_at=result.get("trained_at")
        )
        
    except ValueError as e:
        logger.error(f"Erro de validação no treinamento: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao treinar modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro no treinamento: {str(e)}")


@app.post("/predict-custom", response_model=CustomPredictionResponse)
def predict_custom_data(request: CustomPredictionRequest):
    """
    Endpoint isolado que recebe dados históricos personalizados,
    treina um modelo temporário e retorna previsões.
    
    Este endpoint não salva o modelo e funciona de forma completamente independente.
    O modelo é treinado em memória apenas para esta requisição.
    
    Args:
        historical_data: Lista de pontos históricos com date, close e volume
        days: Número de dias para prever (padrão: 1)
        ticker_name: Nome opcional para identificação (padrão: "CUSTOM")
    
    Returns:
        Previsões, métricas de treinamento e informações do processo
    """
    try:
        import pandas as pd
        import numpy as np
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error
        
        logger.info(f"Recebida requisição de predição customizada: {len(request.historical_data)} pontos, {request.days} dias")
        
        # Iniciar tracking MLFlow
        with mlflow.start_run(run_name=f"predict_custom_{request.ticker_name}_{request.days}days"):
            mlflow.log_param("ticker_name", request.ticker_name)
            mlflow.log_param("days", request.days)
            mlflow.log_param("endpoint", "POST /predict-custom")
            mlflow.log_metric("historical_data_points", len(request.historical_data))
        
            if len(request.historical_data) < 30:
            raise HTTPException(
                status_code=400,
                detail=f"Mínimo de 30 pontos históricos necessários. Fornecidos: {len(request.historical_data)}"
            )
        
        data_dict = {
            'date': [pd.to_datetime(point.date) for point in request.historical_data],
            'close': [point.close for point in request.historical_data],
            'volume': [point.volume for point in request.historical_data]
        }
        df = pd.DataFrame(data_dict)
        df = df.sort_values('date').reset_index(drop=True)
        
        SEQ_LENGTH = 30
        TRAIN_SPLIT = 0.8
        EPOCHS = 50
        LEARNING_RATE = 0.001
        
        X = df[['close', 'volume']].values
        y = df[['close']].values
        
        feature_scaler = MinMaxScaler()
        close_scaler = MinMaxScaler()
        
        X_scaled = feature_scaler.fit_transform(X)
        y_scaled = close_scaler.fit_transform(y)
        
        def create_sequences(X, y, seq_length):
            Xs, ys = [], []
            for i in range(len(X) - seq_length):
                Xs.append(X[i:i + seq_length])
                ys.append(y[i + seq_length])
            return np.array(Xs), np.array(ys)
        
        train_size = int(len(X_scaled) * TRAIN_SPLIT)
        
        X_train_raw = X_scaled[:train_size]
        y_train_raw = y_scaled[:train_size]
        
        X_test_raw = X_scaled[train_size - SEQ_LENGTH:]
        y_test_raw = y_scaled[train_size - SEQ_LENGTH:]
        
        X_train, y_train = create_sequences(X_train_raw, y_train_raw, SEQ_LENGTH)
        X_test, y_test = create_sequences(X_test_raw, y_test_raw, SEQ_LENGTH)
        
        X_train_torch = torch.tensor(X_train, dtype=torch.float32)
        y_train_torch = torch.tensor(y_train, dtype=torch.float32)
        
        X_test_torch = torch.tensor(X_test, dtype=torch.float32)
        y_test_torch = torch.tensor(y_test, dtype=torch.float32)
        
        class LSTMModel(nn.Module):
            def __init__(self, input_size=2, hidden_size=64, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                return self.fc(out)
        
        model = LSTMModel(input_size=2, hidden_size=64, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        logger.info(f"Iniciando treinamento temporário com {len(X_train)} sequências")
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            output = model(X_train_torch)
            loss = criterion(output, y_train_torch)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Época {epoch + 1}/{EPOCHS} - Loss: {loss.item():.6f}")
        
        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_test_torch).numpy()
        
        preds = close_scaler.inverse_transform(preds_scaled)
        reals = close_scaler.inverse_transform(y_test_torch.numpy())
        
        rmse = np.sqrt(mean_squared_error(reals, preds))
        
        # Log métricas de treino no MLFlow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_param("seq_length", SEQ_LENGTH)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        
        logger.info(f"RMSE do modelo temporário: {rmse:.2f}")
        
        predictions = []
        current_data = X_scaled.copy()
        
        for step in range(request.days):
            window = current_data[-SEQ_LENGTH:]
            window = window.reshape(1, SEQ_LENGTH, 2)
            input_tensor = torch.tensor(window, dtype=torch.float32)
            
            with torch.no_grad():
                pred_scaled_value = model(input_tensor).cpu().numpy()[0][0]
            
            pred_price = close_scaler.inverse_transform([[pred_scaled_value]])[0][0]
            predictions.append(float(pred_price))
            
            if step < request.days - 1:
                last_volume_scaled = current_data[-1, 1]
                new_point = np.array([[pred_scaled_value, last_volume_scaled]])
                current_data = np.vstack([current_data, new_point])
        
        last_known_price = float(df['close'].iloc[-1])
        
        # Log das previsões no MLFlow
        mlflow.log_metric("last_known_price", last_known_price)
        for i, pred in enumerate(predictions, 1):
            mlflow.log_metric(f"prediction_day_{i}", pred)
        mlflow.log_param("status", "success")
        
        logger.info(f"Previsão customizada concluída: {request.days} dias")
        
        return CustomPredictionResponse(
            ticker_name=request.ticker_name,
            predictions=[round(p, 2) for p in predictions],
            days=request.days,
            last_known_price=round(last_known_price, 2),
            rmse=round(rmse, 2),
            training_samples=len(X_train),
            message=f"Modelo treinado e previsão realizada com sucesso usando {len(request.historical_data)} pontos históricos"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao processar predição customizada: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
