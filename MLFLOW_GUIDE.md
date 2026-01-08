# Guia MLFlow - Sistema de Previsão de Ações

## 📊 O que é MLFlow?

MLFlow é uma plataforma open-source para gerenciar o ciclo de vida completo de Machine Learning. Neste projeto, ele está **sempre habilitado** e rastreia automaticamente:

- ✅ Parâmetros de treinamento (épocas, learning rate, etc.)
- ✅ Métricas de performance (RMSE, acurácia)
- ✅ Modelos treinados (arquitetura e pesos)
- ✅ Artefatos (scalers, checkpoints)
- ✅ Previsões realizadas
- ✅ Dados de entrada (quantidade de pontos, datas)

## 🚀 Iniciando o MLFlow UI

### 1. Visualizar Experimentos Localmente

```bash
mlflow ui
```

Acesse: http://localhost:5000

### 2. Visualizar Experimentos em Porta Específica

```bash
mlflow ui --port 5001
```

Acesse: http://localhost:5001

### 3. MLFlow UI com Backend Remoto

Se você configurou um servidor MLFlow remoto:

```bash
# No .env
MLFLOW_TRACKING_URI=http://seu-servidor-mlflow:5000

# Iniciar UI
mlflow ui --backend-store-uri http://seu-servidor-mlflow:5000
```

## 📁 Estrutura de Dados do MLFlow

```
mlruns/
├── 0/                          # Experimento padrão
│   ├── meta.yaml
│   └── <run-id>/               # Cada execução tem um ID único
│       ├── artifacts/          # Modelos, scalers salvos
│       ├── metrics/            # RMSE, predictions, etc.
│       ├── params/             # Hiperparâmetros
│       └── tags/               # Tags customizadas
└── <experiment-id>/            # Experimento "stock-price-prediction"
    └── ...
```

## 🔍 O que é Rastreado Automaticamente

### Durante o Treinamento (`/train`)

```python
# Parâmetros
- ticker
- start_date
- end_date
- seq_length: 30
- epochs: 50
- learning_rate: 0.001
- hidden_size: 64
- num_layers: 2

# Métricas
- rmse (train)
- rmse (test)
- next_prediction
- data_points

# Artefatos
- lstm_model_{ticker}.pth
- scaler_features_{ticker}.save
- scaler_close_{ticker}.save
```

### Durante Previsões (`/predict`)

```python
# Parâmetros
- ticker
- days
- endpoint (GET ou POST)
- start_date (opcional)
- end_date (opcional)

# Métricas
- data_points
- last_known_price
- prediction_day_1
- prediction_day_2
- ...
- prediction_day_N
```

### Durante Previsões Customizadas (`/predict-custom`)

```python
# Parâmetros
- ticker_name
- days
- endpoint
- seq_length
- epochs
- learning_rate

# Métricas
- historical_data_points
- train_samples
- test_samples
- rmse
- last_known_price
- prediction_day_1, prediction_day_2, ...
```

## 📊 Visualizando Experimentos

### 1. Interface Web

Após executar `mlflow ui`, você verá:

- **Runs**: Lista de todas as execuções
- **Parameters**: Hiperparâmetros de cada run
- **Metrics**: Gráficos de métricas ao longo do tempo
- **Artifacts**: Modelos e arquivos salvos
- **Comparison**: Comparar múltiplas execuções

### 2. Filtrando Resultados

```python
# Filtrar por ticker
ticker = "VALE3.SA"

# Filtrar por RMSE baixo
metrics.rmse < 5.0

# Filtrar por data
attributes.start_time > "2026-01-01"
```

### 3. Comparando Modelos

No MLFlow UI:
1. Selecione múltiplas runs (checkbox)
2. Clique em "Compare"
3. Visualize gráficos lado a lado
4. Identifique o melhor modelo

## 🔧 Configuração Avançada

### Usar MLFlow com Servidor Remoto

```bash
# .env
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_EXPERIMENT_NAME=stock-production
```

### Usar MLFlow com PostgreSQL

```bash
# .env
MLFLOW_TRACKING_URI=postgresql://user:password@localhost/mlflow
```

### Usar MLFlow com S3 para Artefatos

```bash
# .env
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
AWS_ACCESS_KEY_ID=sua-chave
AWS_SECRET_ACCESS_KEY=seu-secret
```

## 📈 Métricas Importantes

### RMSE (Root Mean Square Error)

Mede o erro médio das previsões:

- **RMSE < 2**: Excelente
- **RMSE 2-5**: Bom
- **RMSE 5-10**: Aceitável
- **RMSE > 10**: Necessita ajustes

### Comparação de Previsões

O MLFlow permite visualizar:
- Previsões vs. Valores Reais
- Tendência ao longo do tempo
- Acurácia por ticker
- Performance por período

## 🎯 Exemplos de Uso

### 1. Treinar e Visualizar

```bash
# Treinar modelo
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA"}'

# Visualizar no MLFlow
mlflow ui
# Abrir http://localhost:5000
# Ver experimento "stock-price-prediction"
# Verificar métricas de RMSE
```

### 2. Comparar Diferentes Tickers

```bash
# Treinar múltiplos tickers
curl -X POST http://localhost:8000/train -d '{"ticker": "PETR4.SA"}'
curl -X POST http://localhost:8000/train -d '{"ticker": "VALE3.SA"}'
curl -X POST http://localhost:8000/train -d '{"ticker": "ITUB4.SA"}'

# No MLFlow UI:
# - Filtrar por ticker
# - Comparar RMSE
# - Identificar melhor performance
```

### 3. Rastrear Previsões ao Longo do Tempo

```bash
# Fazer previsões diárias
curl "http://localhost:8000/predict/PETR4.SA?days=1"

# No MLFlow:
# - Ver histórico de previsões
# - Comparar com preços reais
# - Avaliar acurácia temporal
```

## 🔒 Boas Práticas

### 1. Nomear Experimentos

```python
# Produção
MLFLOW_EXPERIMENT_NAME=stock-production

# Desenvolvimento
MLFLOW_EXPERIMENT_NAME=stock-dev

# Testes
MLFLOW_EXPERIMENT_NAME=stock-experiments
```

### 2. Tags Customizadas

```python
# Em model_training.py ou api.py
mlflow.set_tag("environment", "production")
mlflow.set_tag("model_version", "v2.0")
mlflow.set_tag("data_source", "yahoo_finance")
```

### 3. Backup de Experimentos

```bash
# Exportar experimentos
mlflow experiments export --experiment-id 0 --output-dir backup/

# Importar experimentos
mlflow experiments import --input-dir backup/
```

## 🐛 Troubleshooting

### MLFlow UI não inicia

```bash
# Verificar se a porta está ocupada
lsof -i :5000  # Linux/Mac
netstat -ano | findstr :5000  # Windows

# Usar outra porta
mlflow ui --port 5001
```

### Experimentos não aparecem

```bash
# Verificar diretório mlruns
ls -la mlruns/

# Verificar variável de ambiente
echo $MLFLOW_TRACKING_URI

# Resetar para local
unset MLFLOW_TRACKING_URI
mlflow ui
```

### Erro ao salvar artefatos

```bash
# Verificar permissões
chmod -R 755 mlruns/

# Verificar espaço em disco
df -h
```

## 📚 Recursos Adicionais

- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLFlow Models](https://mlflow.org/docs/latest/models.html)
- [MLFlow Projects](https://mlflow.org/docs/latest/projects.html)

## 🎓 Conclusão

O MLFlow está totalmente integrado ao sistema de previsão de ações:

1. ✅ **Sempre habilitado** - Rastreamento automático em todas as operações
2. ✅ **Completo** - Rastreia treinamento, previsões e métricas
3. ✅ **Transparente** - Não interfere no funcionamento da API
4. ✅ **Valioso** - Facilita comparações e melhorias do modelo

Use o MLFlow para:
- Comparar performance entre tickers
- Otimizar hiperparâmetros
- Rastrear previsões ao longo do tempo
- Identificar modelos que precisam retreinamento
- Documentar experimentos

**Comando principal:**
```bash
mlflow ui
```

Acesse http://localhost:5000 e explore seus experimentos! 🚀
