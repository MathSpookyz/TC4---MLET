# 1. Importação das bibliotecas
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# 2. Download dos dados (exemplo com PETR4.SA)
ticker = input("Digite o código da ação: ")  # Exemplo: PETR4.SA
data = yf.download(ticker, start="2020-01-01", end="2025-11-19")['Close'].values.reshape(-1,1)

# 3. Normalização dos dados
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 4. Preparação das sequências (janela deslizante)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)
seq_length = 30
X, y = create_sequences(data_scaled, seq_length)

# 5. Conversão para tensores PyTorch
X_torch = torch.from_numpy(X).float()
y_torch = torch.from_numpy(y).float()

# 6. Definição do modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # pega a saída do último passo de tempo
        out = self.fc(out)
        return out

model = LSTMModel()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. Treinamento do modelo
epochs = 50
for epoch in range(epochs):
    model.train()
    output = model(X_torch)
    loss = loss_function(output, y_torch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%10==0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 8. Previsão (Inferência)
model.eval()
last_seq = torch.from_numpy(data_scaled[-seq_length:]).float().unsqueeze(0)  # batch de 1
with torch.no_grad():
    prediction = model(last_seq).item()
    predicted_price = scaler.inverse_transform([[prediction]])[0][0]
    print(f"Preço previsto da próxima janela para {ticker}: R$ {predicted_price:.2f}")
    
# ===== BACKTEST SIMPLES =====
model.eval()

# Separar dados
test_seq = data_scaled[-(seq_length+1):-1]  # últimos 30 dias
real_price = data[-1][0]  # preço real conhecido

test_seq = torch.from_numpy(test_seq).float().unsqueeze(0)

with torch.no_grad():
    pred_norm = model(test_seq).item()
    pred_price = scaler.inverse_transform([[pred_norm]])[0][0]

print(f"\nBacktest:")
print(f"Preço real: R$ {real_price:.2f}")
print(f"Preço previsto: R$ {pred_price:.2f}")
print(f"Erro absoluto: R$ {abs(real_price - pred_price):.2f}")

