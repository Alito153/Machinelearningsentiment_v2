import pandas as pd

df = pd.read_csv('/app/forex_macro_sentiment_1329/data/macro_events_labeled.csv')
print("All columns:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")