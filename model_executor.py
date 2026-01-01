import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd


SEQ_LENGTH = 30


class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def predict_price(df_processed: pd.DataFrame, ticker: str = None, days: int = 1) -> dict:
    model = LSTMModel()
    model.load_state_dict(torch.load("export/lstm_model.pth", map_location="cpu"))
    model.eval()

    scaler_features = joblib.load("export/scaler_features.save")
    scaler_close = joblib.load("export/scaler_close.save")
    
    if ticker and 'ticker' in df_processed.columns:
        df_processed = df_processed[df_processed['ticker'] == ticker].copy()
    
    if 'date' in df_processed.columns:
        df_processed = df_processed.sort_values('date')
    else:
        df_processed = df_processed.sort_index()
    
    if len(df_processed) < SEQ_LENGTH:
        raise ValueError(f"Dados insuficientes. Necessário pelo menos {SEQ_LENGTH} registros, fornecido {len(df_processed)}")
    
    predictions = []
    current_data = df_processed[['close', 'volume']].values.copy()
    
    for day in range(days):
        last_sequence = current_data[-SEQ_LENGTH:]
        last_sequence_scaled = scaler_features.transform(last_sequence)
        tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_norm = model(tensor).item()
        
        pred_price = scaler_close.inverse_transform([[pred_norm]])[0][0]
        predictions.append(pred_price)
        
        last_volume = current_data[-1, 1]
        new_row = np.array([[pred_price, last_volume]])
        current_data = np.vstack([current_data, new_row])
    
    return {
        'predictions': predictions,
        'days': days,
        'last_known_price': float(df_processed['close'].iloc[-1])
    }
