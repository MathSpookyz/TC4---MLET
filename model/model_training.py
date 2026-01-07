import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import sys
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_ENABLED = True
except ImportError:
    MLFLOW_ENABLED = False
    print("MLFlow não instalado. Instale com: pip install mlflow")

sys.path.append(str(Path(__file__).parent.parent / "scrapper"))

from scrapper_pipeline import get_or_scrappe_ticker


SEQ_LENGTH = 30
TRAIN_SPLIT = 0.8
EPOCHS = 50
LEARNING_RATE = 0.001

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_DIR = os.path.join(BASE_DIR, "export")
os.makedirs(EXPORT_DIR, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "stock-price-prediction")

if MLFLOW_ENABLED:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)


def train_model(ticker: str, start_date: str = "2020-01-01", end_date: str = None) -> dict:
    """
    Treina um modelo LSTM para um ticker específico.
    
    Args:
        ticker: Código da ação (ex: PETR4.SA, VALE3.SA)
        start_date: Data inicial para buscar dados
        end_date: Data final para buscar dados (opcional, padrão: hoje)
    
    Returns:
        dict com informações do treinamento e métricas
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    ticker = ticker.upper()
    
    if MLFLOW_ENABLED:
        mlflow.start_run(run_name=f"train_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)
        mlflow.log_param("seq_length", SEQ_LENGTH)
        mlflow.log_param("train_split", TRAIN_SPLIT)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("hidden_size", 64)
        mlflow.log_param("num_layers", 2)
    
    try:
        print(f"\n{'='*50}")
        print(f"Iniciando treinamento para {ticker}")
        print(f"{'='*50}\n")
        
        
        print("Obtendo dados históricos...")
        df = get_or_scrappe_ticker(
            ticker=ticker,
            data_path="scrapper/data/processed/prices",
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            raise ValueError(f"Nenhum dado retornado para o ticker {ticker}")
        
        df = df[df["ticker"] == ticker].copy()
        df = df[["close", "volume"]].dropna()
        
        print(f"Dados obtidos: {len(df)} registros de {df.index.min()} até {df.index.max()}")
        
        if MLFLOW_ENABLED:
            mlflow.log_metric("dataset_size", len(df))
        
        
        X = df[["close", "volume"]].values
        y = df[["close"]].values
        
        
        feature_scaler = MinMaxScaler()
        close_scaler = MinMaxScaler()
        
        X_scaled = feature_scaler.fit_transform(X)
        y_scaled = close_scaler.fit_transform(y)
        
        
        train_size = int(len(X_scaled) * TRAIN_SPLIT)
        
        X_train_raw = X_scaled[:train_size]
        y_train_raw = y_scaled[:train_size]
        
        X_test_raw = X_scaled[train_size - SEQ_LENGTH:]
        y_test_raw = y_scaled[train_size - SEQ_LENGTH:]
        
        X_train, y_train = create_sequences(X_train_raw, y_train_raw, SEQ_LENGTH)
        X_test, y_test = create_sequences(X_test_raw, y_test_raw, SEQ_LENGTH)
        
        print(f"Dados de treino: {len(X_train)} sequências")
        print(f"Dados de teste: {len(X_test)} sequências")
        
        if MLFLOW_ENABLED:
            mlflow.log_metric("train_sequences", len(X_train))
            mlflow.log_metric("test_sequences", len(X_test))
        
        
        X_train_torch = torch.tensor(X_train, dtype=torch.float32)
        y_train_torch = torch.tensor(y_train, dtype=torch.float32)
        
        X_test_torch = torch.tensor(X_test, dtype=torch.float32)
        y_test_torch = torch.tensor(y_test, dtype=torch.float32)
        
        
        model = LSTMModel(
            input_size=2,
            hidden_size=64,
            num_layers=2
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        
        print(f"\nIniciando treinamento ({EPOCHS} épocas)...")
        
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            
            output = model(X_train_torch)
            loss = criterion(output, y_train_torch)
            
            loss.backward()
            optimizer.step()
            
            if MLFLOW_ENABLED:
                mlflow.log_metric("train_loss", loss.item(), step=epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f"Época {epoch + 1}/{EPOCHS} - Loss: {loss.item():.6f}")
        
        
        print("\nExecutando backtest...")
        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_test_torch).numpy()
        
        preds = close_scaler.inverse_transform(preds_scaled)
        reals = close_scaler.inverse_transform(y_test_torch.numpy())
        
        rmse = np.sqrt(mean_squared_error(reals, preds))
        mae = np.mean(np.abs(reals - preds))
        mape = np.mean(np.abs((reals - preds) / reals)) * 100
        
        print("\n===== RESULTADOS DO BACKTEST =====")
        print(f"Ticker: {ticker}")
        print(f"RMSE: R$ {rmse:.2f}")
        print(f"MAE: R$ {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Preço real último dia: R$ {reals[-1][0]:.2f}")
        print(f"Preço previsto último dia: R$ {preds[-1][0]:.2f}")
        print(f"Erro absoluto: R$ {abs(reals[-1][0] - preds[-1][0]):.2f}")
        
        if MLFLOW_ENABLED:
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mape", mape)
            mlflow.log_metric("last_real_price", float(reals[-1][0]))
            mlflow.log_metric("last_predicted_price", float(preds[-1][0]))
            mlflow.log_metric("last_absolute_error", float(abs(reals[-1][0] - preds[-1][0])))
        
        
        last_sequence = X_scaled[-SEQ_LENGTH:]
        last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            next_scaled = model(last_sequence).item()
        
        next_price = close_scaler.inverse_transform([[next_scaled]])[0][0]
        print(f"\nPreço previsto próximo dia ({ticker}): R$ {next_price:.2f}")
        
        if MLFLOW_ENABLED:
            mlflow.log_metric("next_day_prediction", next_price)
        
        
        MODEL_PATH = os.path.join(EXPORT_DIR, f"lstm_model_{ticker}.pth")
        SCALER_FEATURES_PATH = os.path.join(EXPORT_DIR, f"scaler_features_{ticker}.save")
        SCALER_CLOSE_PATH = os.path.join(EXPORT_DIR, f"scaler_close_{ticker}.save")
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_size": 2,
                "hidden_size": 64,
                "num_layers": 2,
                "seq_length": SEQ_LENGTH
            },
            "version": "v1",
            "trained_on": ticker,
            "trained_at": datetime.utcnow().isoformat(),
            "metrics": {
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "last_real_price": float(reals[-1][0]),
                "last_predicted_price": float(preds[-1][0]),
                "next_prediction": float(next_price)
            }
        }
        
        torch.save(checkpoint, MODEL_PATH)
        joblib.dump(feature_scaler, SCALER_FEATURES_PATH)
        joblib.dump(close_scaler, SCALER_CLOSE_PATH)
        
        print(f"\n✓ Modelo salvo em: {MODEL_PATH}")
        print(f"✓ Scaler features salvo em: {SCALER_FEATURES_PATH}")
        print(f"✓ Scaler close salvo em: {SCALER_CLOSE_PATH}")
        
        if MLFLOW_ENABLED:
            mlflow.pytorch.log_model(model, "model")
            
            mlflow.log_artifact(MODEL_PATH, "checkpoints")
            mlflow.log_artifact(SCALER_FEATURES_PATH, "scalers")
            mlflow.log_artifact(SCALER_CLOSE_PATH, "scalers")
            
            mlflow.set_tag("ticker", ticker)
            mlflow.set_tag("model_type", "LSTM")
            mlflow.set_tag("framework", "PyTorch")
        
        print(f"\n{'='*50}")
        print("Treinamento concluído com sucesso!")
        print(f"{'='*50}\n")
        
        result = {
            "ticker": ticker,
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "last_real_price": float(reals[-1][0]),
            "last_predicted_price": float(preds[-1][0]),
            "next_prediction": float(next_price),
            "model_path": MODEL_PATH,
            "trained_at": checkpoint["trained_at"]
        }
        
        if MLFLOW_ENABLED:
            mlflow.end_run()
        
        return result
        
    except Exception as e:
        if MLFLOW_ENABLED:
            mlflow.end_run(status="FAILED")
        raise e



if __name__ == "__main__":
    ticker = input("Digite o código da ação (ex: PETR4.SA, VALE3.SA, ITUB4.SA): ").strip()
    
    if not ticker:
        print("Ticker não pode ser vazio!")
        exit(1)
    
    try:
        result = train_model(ticker)
        print("\nResumo do treinamento:")
        print(f"  - RMSE: R$ {result['rmse']:.2f}")
        print(f"  - Próxima previsão: R$ {result['next_prediction']:.2f}")
    except Exception as e:
        print(f"\nErro durante o treinamento: {e}")
        exit(1)
