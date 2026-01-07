from scrapper.scrapper_pipeline import get_or_scrappe_ticker

df = get_or_scrappe_ticker("PETR4.SA")
print(df.head())
