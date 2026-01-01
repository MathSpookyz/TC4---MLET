from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "scapper"))

from model_executor import predict_price
from scrapper_pipeline import get_or_scrappe_ticker

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


class PredictionRequest(BaseModel):
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    days: Optional[int] = 1


class PredictionResponse(BaseModel):
    ticker: str
    predictions: list[float]
    days: int
    last_known_price: float
    currency: str = "BRL"


@app.get("/")
def root():
    return {
        "message": "Stock Price Prediction API",
        "endpoints": {
            "predict": "/predict/{ticker}",
            "predict_post": "/predict",
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
        
        df_processed = get_or_scrappe_ticker(
            ticker=request.ticker.upper(),
            data_path="scapper/data/processed/prices",
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if df_processed is None or df_processed.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Não foi possível obter dados para o ticker {request.ticker}"
            )
        
        result = predict_price(df_processed, ticker=request.ticker.upper(), days=request.days)
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
