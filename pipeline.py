from scr.extract.yahoo_extractor import extract_prices
from scr.transform.price_transformer import tratar_prices
from scr.load.parquet_loader import (
    save_raw_parquet,
    save_processed_parquet,
    parquet_exists
)

RAW_PATH = "data/raw/prices_raw.parquet"
PROCESSED_PATH = "data/processed/prices/"

def main():
    tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
    start_date = "2019-01-01"
    end_date = "2024-01-01"

    # =========================
    # RAW (Cache de extração)
    # =========================
    if not parquet_exists(RAW_PATH):
        print("🔄 Extraindo dados do Yahoo Finance...")
        df_raw = extract_prices(tickers, start_date, end_date)
        save_raw_parquet(df_raw, RAW_PATH)
    else:
        print("✅ Cache RAW encontrado")

        df_raw = extract_prices(tickers, start_date, end_date)
        # (em produção você leria o parquet, aqui mantemos simples)

    # =========================
    # TRANSFORM
    # =========================
    print("⚙️ Tratando dados...")
    df_processed = tratar_prices(df_raw)

    # =========================
    # LOAD (Cache escalável)
    # =========================
    print("💾 Salvando dados tratados em Parquet...")
    save_processed_parquet(df_processed, PROCESSED_PATH)

    print("🚀 Pipeline finalizado com sucesso!")

if __name__ == "__main__":
    main()