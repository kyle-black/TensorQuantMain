import pandas as pd


df = pd.read_json('updated_data/EURUSD.json')


df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].apply(lambda x: x.timestamp())
# Rename columns
df.rename(columns={'date': 'Date', 'open': 'Open','high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)

#print(df)
# Set 'Date' as the index
df.set_index('Date', inplace=True)

print(df)

#df = pd.read_json('updated_data/EURUSD.json')
df.to_csv('updated_data/EURUSD.csv')