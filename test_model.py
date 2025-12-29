import torch
import torch.nn as nn
import numpy as np
import joblib
import yfinance as yf


TICKER = "PETR4.SA"
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


model = LSTMModel()
model.load_state_dict(torch.load("lstm_model.pth", map_location="cpu"))
model.eval()

scaler_features = joblib.load("scaler_features.save")
scaler_close = joblib.load("scaler_close.save")

print("Modelo e scalers carregados!")


data = yf.download(TICKER, period="3mo")
features = data[['Close', 'Volume']].values


last_sequence = features[-SEQ_LENGTH:]


last_sequence_scaled = scaler_features.transform(last_sequence)


tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0)


with torch.no_grad():
    pred_norm = model(tensor).item()

pred_price = scaler_close.inverse_transform([[pred_norm]])[0][0]

print(f"\nPreço previsto para o próximo dia ({TICKER}): R$ {pred_price:.2f}")
