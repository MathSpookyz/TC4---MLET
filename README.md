# API de Previsão de Preços de Ações

Sistema completo de previsão de preços de ações usando modelo LSTM (Long Short-Term Memory) com pipeline ETL automatizado, API REST e monitoramento MLFlow.

## 🚀 Características Principais

- ✅ **Multi-ticker**: Suporta qualquer ação da bolsa brasileira
- ✅ **Treinamento automático**: Treina modelos sob demanda quando necessário
- ✅ **Armazenamento híbrido**: Local ou S3 via variável de ambiente
- ✅ **MLFlow**: Monitoramento completo de experimentos
- ✅ **Dados personalizados**: Endpoint para treinar/prever com seus próprios dados
- ✅ **Logs detalhados**: Rastreamento completo de operações
- ✅ **Cache inteligente**: Modelos permanecem em memória após carregamento

## 📋 Requisitos

- Python 3.11+
- Docker (opcional)

## ⚡ Início Rápido

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar Ambiente (Opcional)

Crie `.env` na raiz do projeto:

```bash
# Armazenamento (padrão: local)
STORAGE_TYPE=local

# Para usar S3:
# STORAGE_TYPE=s3
# S3_BUCKET=seu-bucket
# AWS_ACCESS_KEY_ID=sua-chave
# AWS_SECRET_ACCESS_KEY=seu-secret

# MLFlow (padrão: local)
MLFLOW_TRACKING_URI=file:./mlruns
```

### 3. Iniciar API

```bash
python api.py
```

Acesse: http://localhost:8000/docs

## 📊 Como Usar

### Treinar um Modelo

#### Via Linha de Comando

```bash
python model/model_training.py
```

O sistema solicitará o ticker (ex: PETR4.SA, VALE3.SA, ITUB4.SA)

#### Via API

```bash
# Treinar modelo via API
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"ticker": "VALE3.SA", "start_date": "2020-01-01"}'
```

### Fazer Previsões

#### Previsão Simples (com treinamento automático)

```bash
# O sistema treina automaticamente se o modelo não existir
curl "http://localhost:8000/predict/PETR4.SA?days=7"
```

#### Via API - GET

```bash
# Previsão de 1 dia para VALE3.SA
curl http://localhost:8000/predict/VALE3.SA

# Previsão de 7 dias
curl "http://localhost:8000/predict/VALE3.SA?days=7"

# Com período específico
curl "http://localhost:8000/predict/ITUB4.SA?days=5&start_date=2023-01-01&end_date=2024-12-31"
```

#### Via API - POST

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "PETR4.SA",
    "days": 30,
    "start_date": "2023-01-01",
    "end_date": "2024-12-31"
  }'
```

### Estrutura de Resposta

#### Treinamento

```json
{
  "ticker": "VALE3.SA",
  "status": "success",
  "message": "Modelo treinado com sucesso para VALE3.SA",
  "rmse": 2.45,
  "next_prediction": 65.32,
  "trained_at": "2026-01-07T10:30:00.000000"
}
```

#### Previsão

```json
{
  "ticker": "VALE3.SA",
  "predictions": [65.32, 65.87, 66.15],
  "days": 3,
  "last_known_price": 64.80,
  "currency": "BRL"
}
```

### Usar Dados Personalizados

O endpoint `/predict-custom` permite treinar um modelo temporário com seus próprios dados históricos. Ideal para:

- Testar estratégias com dados históricos específicos
- Fazer previsões com dados sintéticos ou simulados
- Validar o modelo sem interferir com modelos salvos
- Treinar e prever sem dependência de tickers da bolsa

#### Características do Endpoint Custom

✅ **Isolado**: Não salva o modelo no disco  
✅ **Temporário**: Modelo existe apenas durante a requisição  
✅ **Flexível**: Aceita qualquer conjunto de dados históricos  
✅ **Completo**: Retorna métricas de treinamento (RMSE)

#### Requisitos de Dados

- **Mínimo**: 30 pontos históricos
- **Recomendado**: 60+ pontos para melhor acurácia
- **Formato**: JSON com date, close e volume

#### Exemplo de Uso

**Recomendado para Windows PowerShell:**
```powershell
# Usar arquivo JSON (evita problemas com formatação)
curl -X POST http://localhost:8000/predict-custom -H "Content-Type: application/json" -d "@example_custom_data.json"
```

**Para Linux/Mac/Git Bash:**
```bash
# Usar arquivo JSON (mais simples)
curl -X POST http://localhost:8000/predict-custom \
  -H "Content-Type: application/json" \
  -d @example_custom_data.json

# OU com heredoc (dados inline)
curl -X POST http://localhost:8000/predict-custom \
  -H "Content-Type: application/json" \
  -d @- << 'EOF'
{
  "ticker_name": "TESTE",
  "days": 5,
  "historical_data": [
    {"date": "2024-01-01", "close": 100.5, "volume": 1000000},
    {"date": "2024-01-02", "close": 101.2, "volume": 1100000}
  ]
}
EOF
```

**Nota:** O arquivo `example_custom_data.json` incluso no projeto já tem 45 pontos de dados prontos para teste.

**Resposta esperada:**
```json
{
  "ticker_name": "TESTE_RAPIDO",
  "predictions": [118.45, 119.12, 119.78],
  "days": 3,
  "last_known_price": 117.90,
  "rmse": 0.87,
  "training_samples": 0,
  "message": "Modelo treinado e previsão realizada com sucesso usando 30 pontos históricos"
}
```

#### Estrutura dos Dados

Cada ponto histórico deve conter:

```json
{
  "date": "YYYY-MM-DD",
  "close": float,
  "volume": int
}
```

#### Parâmetros

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `ticker_name` | string | Não | Nome para identificação (padrão: "CUSTOM") |
| `days` | integer | Não | Dias para prever (padrão: 1) |
| `historical_data` | array | Sim | Lista de pontos históricos (mínimo 30) |

#### Resposta do Endpoint Custom

```json
{
  "ticker_name": "MINHA_ACAO",
  "predictions": [102.45, 103.12, 103.78, 104.21, 104.67],
  "days": 5,
  "last_known_price": 101.80,
  "rmse": 1.23,
  "training_samples": 120,
  "message": "Modelo treinado e previsão realizada com sucesso usando 150 pontos históricos"
}
```

#### Campos da Resposta

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `ticker_name` | string | Nome fornecido na requisição |
| `predictions` | array | Lista de previsões (um valor por dia) |
| `days` | integer | Número de dias previstos |
| `last_known_price` | float | Último preço real conhecido |
| `rmse` | float | Root Mean Square Error do modelo |
| `training_samples` | integer | Número de sequências usadas no treino |
| `message` | string | Mensagem descritiva |

#### Erros Comuns

**400 - Dados Insuficientes**
```json
{"detail": "Mínimo de 30 pontos históricos necessários. Fornecidos: 20"}
```
Solução: Fornecer pelo menos 30 pontos de dados históricos.

**400 - Formato Inválido**
```json
{"detail": "Campo 'close' inválido no ponto 5"}
```
Solução: Verificar se todos os pontos têm date, close e volume válidos.

#### Casos de Uso do Endpoint Custom

1. **Validação de Estratégias**: Teste sua estratégia de trading com dados históricos específicos
2. **Testes com Dados Sintéticos**: Valide o modelo com dados gerados
3. **Análise What-If**: "E se os preços tivessem evoluído diferente?"
4. **Backtesting**: Teste o modelo com períodos históricos específicos

#### Performance do Endpoint Custom

| Métrica | Valor Típico |
|---------|--------------|
| Tempo (50 pontos) | 10-20 segundos |
| Tempo (100 pontos) | 20-40 segundos |
| Tempo (200 pontos) | 40-80 segundos |
| Memória utilizada | ~500 MB |

## 🎯 Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Informações da API |
| GET | `/health` | Health check |
| GET | `/predict/{ticker}` | Previsão para ticker (params: days, start_date, end_date) |
| POST | `/predict` | Previsão com JSON completo |
| POST | `/train` | Treinar modelo para ticker |
| POST | `/predict-custom` | Treinar/prever com dados personalizados (isolado) |

### Comparação de Endpoints

| Característica | `/predict/{ticker}` | `/train` | `/predict-custom` |
|----------------|---------------------|----------|-------------------|
| Usa dados do Yahoo Finance | ✅ | ✅ | ❌ |
| Salva modelo | ❌ | ✅ | ❌ |
| Dados personalizados | ❌ | ❌ | ✅ |
| Cache de modelos | ✅ | ✅ | ❌ |
| Retorna RMSE | ❌ | ✅ | ✅ |
| Isolado | ❌ | ❌ | ✅ |

## 🏗️ Arquitetura do Sistema

### Componentes Principais

**Pipeline ETL**
- `yahoo_extractor.py` - Extração de dados do Yahoo Finance
- `price_transformer.py` - Feature engineering (médias móveis, volatilidade, retornos)
- `parquet_loader.py` - Armazenamento híbrido (local/S3)
- `scrapper_pipeline.py` - Orquestração do pipeline

**Machine Learning**
- `model_training.py` - Treinamento LSTM com MLFlow
- `model_executor.py` - Carregamento e inferência de modelos

**API REST**
- `api.py` - FastAPI com endpoints de previsão e treinamento

### Fluxo de Dados

```
Cliente → API → Verifica Cache Local → Se não existe → Yahoo Finance
                     ↓                                        ↓
                 Model LSTM ← Dados Processados ← Feature Engineering
                     ↓
                 Previsão → Resposta JSON → Cliente
```

### Modelo LSTM

- **Arquitetura**: 2 camadas LSTM, 64 neurônios por camada
- **Features**: Preço de fechamento + Volume
- **Janela temporal**: 30 dias
- **Normalização**: MinMaxScaler
- **Armazenamento**: `export/lstm_model_{TICKER}.pth`

## 📦 Estrutura do Projeto

```
├── api.py                          # API REST FastAPI
├── model_executor.py               # Inferência de modelos
├── requirements.txt                # Dependências Python
├── docker-compose.yml              # Orquestração Docker
├── model/
│   └── model_training.py          # Treinamento LSTM + MLFlow
├── scrapper/
│   ├── scrapper_pipeline.py       # Orquestração ETL
│   ├── scr/
│   │   ├── extract/
│   │   │   └── yahoo_extractor.py # Extração Yahoo Finance
│   │   ├── transform/
│   │   │   └── price_transformer.py # Feature engineering
│   │   └── load/
│   │       └── parquet_loader.py  # Armazenamento local/S3
│   └── data/                      # Dados locais (raw + processed)
└── export/                        # Modelos treinados e scalers
```

## 🐳 Docker

### Executar com Docker Compose (Recomendado)

```bash
# Iniciar
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar
docker-compose down
```

### Docker Standalone

```bash
# Build
docker build -t stock-api .

# Run
docker run -d -p 8000:8000 stock-api
```

## 📈 MLFlow - Monitoramento de Experimentos

O sistema usa MLFlow para rastrear todos os treinamentos.

### Visualizar Experimentos

```bash
# Iniciar MLFlow UI
mlflow ui

# Acessar em: http://localhost:5000
```

**Métricas rastreadas:**
- **Parâmetros**: ticker, datas, épocas, learning rate, batch size
- **Métricas**: RMSE, loss por época, última previsão
- **Artefatos**: Modelos salvos (`.pth`), scalers (`.save`)
- **Tags**: versão, timestamp, duração do treinamento

### Comparar Modelos

No MLFlow UI você pode:
- Comparar RMSE entre diferentes tickers
- Ver evolução do loss durante treinamento
- Analisar distribuição de previsões
- Baixar modelos de versões anteriores
- Filtrar experimentos por parâmetros
- Exportar resultados para análise

## 💡 Exemplos Práticos

### Exemplo 1: Treinar Múltiplos Tickers

```bash
# Treinar PETR4
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA"}'

# Treinar VALE3
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"ticker": "VALE3.SA"}'

# Treinar ITUB4
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"ticker": "ITUB4.SA"}'
```

### Exemplo 2: Previsões para Múltiplos Tickers

```bash
# PETR4 - 7 dias
curl "http://localhost:8000/predict/PETR4.SA?days=7"

# VALE3 - 30 dias
curl "http://localhost:8000/predict/VALE3.SA?days=30"

# ITUB4 - 14 dias
curl "http://localhost:8000/predict/ITUB4.SA?days=14"
```

### Exemplo 3: Dados Personalizados

```bash
# Criar arquivo com dados históricos
cat > custom_data.json << 'EOF'
{
  "ticker_name": "TESTE_ACAO",
  "days": 7,
  "historical_data": [
    {"date": "2024-01-01", "close": 50.00, "volume": 1000000},
    {"date": "2024-01-02", "close": 50.50, "volume": 1050000},
    {"date": "2024-01-03", "close": 51.20, "volume": 1100000},
    ... (mínimo 30 pontos)
  ]
}
EOF

# Fazer previsão com dados personalizados
curl -X POST http://localhost:8000/predict-custom \
  -H "Content-Type: application/json" \
  -d @custom_data.json
```

**Vantagens do endpoint custom:**
- Não precisa ter o ticker na bolsa
- Útil para testes com dados sintéticos
- Não interfere com modelos salvos
- Permite validação de estratégias com dados históricos específicos

Veja também:
- [example_custom_prediction.py](example_custom_prediction.py) - Script completo de exemplo
- [example_custom_data.json](example_custom_data.json) - Dados de exemplo prontos (45 pontos históricos)

**Scripts incluídos:**
```bash
# Script completo com geração automática de dados
python example_custom_prediction.py

# Usar dados do arquivo JSON de exemplo
curl -X POST http://localhost:8000/predict-custom \
  -H "Content-Type: application/json" \
  -d @example_custom_data.json
```

### Exemplo 4: Treinamento Automático

```bash
# Execute o script de demonstração
python example_auto_train.py
```

### Exemplo 5: Script Python para Múltiplos Tickers

```python
import requests

tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']
for ticker in tickers:
    r = requests.post('http://localhost:8000/train', 
                      json={'ticker': ticker})
    print(f'{ticker}: {r.json()}')
```

## 🔧 Troubleshooting

### Erro: "Modelo não encontrado"
- **Solução Automática**: O sistema agora treina automaticamente modelos inexistentes
- **Manual**: Você ainda pode treinar explicitamente com `POST /train` se preferir

### Erro: "Dados insuficientes"
- **Solução**: Ajustar `start_date` para obter mais histórico (mínimo 30 registros)

### Erro: "Ticker não encontrado"
- **Solução**: Verificar se o ticker está correto (ex: PETR4.SA, não PETR4)

### Erro: "Mínimo de 30 pontos históricos necessários"
- **Solução**: Para `/predict-custom`, fornecer pelo menos 30 pontos de dados históricos
- **Recomendação**: Use 60+ pontos para melhor acurácia do modelo

### Treinamento Muito Lento (Endpoint Custom)
- **Solução 1**: Reduza o número de pontos históricos
- **Solução 2**: Verifique recursos do servidor (CPU/RAM)
- **Solução 3**: Aumente o timeout do cliente (padrão: 300s)

### RMSE Muito Alto
- **Solução 1**: Verifique a qualidade dos dados
- **Solução 2**: Aumente o número de pontos históricos
- **Solução 3**: Verifique se há valores ausentes ou inconsistentes

### Erro: ModuleNotFoundError
- **Solução**: Verificar se todas as dependências foram instaladas com `pip install -r requirements.txt`

### API não responde
- **Solução**: Verificar se a porta 8000 está disponível e o serviço está rodando

### Ver Logs Detalhados

O sistema possui logs detalhados em todos os processos:

```python
import logging

# Configurar nível de log
logging.basicConfig(
    level=logging.DEBUG,  # Para logs muito detalhados
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

# Rodar a aplicação
# Você verá logs de:
# - Carregamento de modelos
# - Verificação de arquivos
# - Treinamento automático
# - Preparação de dados
# - Previsões
```

## 📊 Performance

| Métrica | Valor |
|---------|-------|
| Tempo com cache | 2-5 segundos |
| Tempo sem cache (1ª vez) | 10-20 segundos |
| Tamanho Docker image | ~2-3GB |
| Memória requerida | ~2GB RAM |

## 🎓 Tickers Suportados

Qualquer ticker do Yahoo Finance (formato: CODIGO.SA para Brasil):

- **Bancos**: ITUB4.SA, BBDC4.SA, SANB11.SA
- **Energia**: PETR4.SA, ELET3.SA
- **Mineração**: VALE3.SA
- **Varejo**: MGLU3.SA, LREN3.SA
- **E muito mais...**

## 📚 Documentação Adicional

- **Swagger UI** - http://localhost:8000/docs (documentação interativa completa)
- **ReDoc** - http://localhost:8000/redoc (documentação alternativa)
- **MLFlow UI** - http://localhost:5000 (após executar `mlflow ui`)

### Documentação da API

A API possui documentação interativa automática gerada pelo FastAPI:

1. **Swagger UI**: Interface interativa onde você pode testar todos os endpoints
   - Acesse: http://localhost:8000/docs
   - Recursos: Teste de endpoints, visualização de schemas, exemplos

2. **ReDoc**: Documentação alternativa em formato de página única
   - Acesse: http://localhost:8000/redoc
   - Recursos: Visualização limpa, navegação fácil, download de spec OpenAPI

## ⚙️ Arquivos de Exemplo

- **example_auto_train.py** - Demonstração de treinamento automático
- **example_custom_prediction.py** - Exemplo completo de uso do endpoint custom
- **example_custom_data.json** - Dados de exemplo prontos para uso
- **test_system.py** - Script de teste completo do sistema

## 🔗 Links Úteis

- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)
- [MLFlow](https://mlflow.org/)
- [Yahoo Finance](https://finance.yahoo.com/)

## 📝 Notas Importantes

1. ✅ **Treinamento automático** - O sistema detecta automaticamente quando um modelo não existe e treina sob demanda
2. 📦 **Modelos independentes** - Cada ticker tem seu próprio modelo (`export/lstm_model_{TICKER}.pth`)
3. 💾 **Cache local** - Dados são salvos localmente para evitar downloads repetidos do Yahoo Finance
4. 🔍 **Logs detalhados** - Todo o processo é logado para fácil debugging
5. ⚙️ **Configuração flexível** - Use variáveis de ambiente para alternar entre local/S3
6. 🚀 **Pronto para produção** - Suporte completo para Docker e MLFlow
7. ⏱️ **Tempo de treinamento** - O treinamento leva geralmente 2-5 minutos por ticker

## 🔄 Fluxo de Trabalho Recomendado

### Para um Novo Ticker

1. **Simplesmente faça a previsão**:
   ```bash
   curl "http://localhost:8000/predict/NOVO_TICKER.SA?days=7"
   ```
   O sistema irá automaticamente:
   - Buscar dados do Yahoo Finance
   - Treinar o modelo
   - Fazer a previsão

2. **Ou treine explicitamente** (opcional):
   ```bash
   curl -X POST http://localhost:8000/train \
     -H "Content-Type: application/json" \
     -d '{"ticker": "NOVO_TICKER.SA"}'
   ```

### Para Retreinar um Ticker Existente

Simplesmente execute o treinamento novamente. O novo modelo substituirá o anterior:

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA", "start_date": "2020-01-01"}'
```

---

**Desenvolvido para FIAP - Sistema de Previsão de Ações com Machine Learning**