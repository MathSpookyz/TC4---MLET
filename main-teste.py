import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

ticker = input("Digite o código da ação (ex: PETR4.SA): ")

data = yf.download(ticker, start="2020-01-01", end="2025-11-19")

features = data[['Close', 'Volume']].values

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

SEQ_LENGTH = 30

train_size = int(len(features_scaled) * 0.8)

train_data = features_scaled[:train_size]
test_data  = features_scaled[train_size - SEQ_LENGTH:]

X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
X_test,  y_test  = create_sequences(test_data,  SEQ_LENGTH)

X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

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
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    output = model(X_train_torch)
    loss = criterion(output, y_train_torch)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    predictions_norm = model(X_test_torch).numpy()


close_scaler = MinMaxScaler()
close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]

predictions = close_scaler.inverse_transform(predictions_norm)
real_values = close_scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(real_values, predictions))

print("\n===== BACKTEST =====")
print(f"RMSE: R$ {rmse:.2f}")
print(f"Preço real último dia: R$ {real_values[-1][0]:.2f}")
print(f"Preço previsto último dia: R$ {predictions[-1][0]:.2f}")
print(f"Erro absoluto: R$ {abs(real_values[-1][0] - predictions[-1][0]):.2f}")

last_sequence = features_scaled[-SEQ_LENGTH:]
last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    next_pred_norm = model(last_sequence).item()

next_price = close_scaler.inverse_transform([[next_pred_norm]])[0][0]

print(f"\nPreço previsto para o próximo dia ({ticker}): R$ {next_price:.2f}")

torch.save(model.state_dict(), "export/lstm_model.pth")


joblib.dump(scaler, "export/scaler_features.save")
joblib.dump(close_scaler, "export/scaler_close.save")

print("\nModelo e scalers exportados com sucesso!")