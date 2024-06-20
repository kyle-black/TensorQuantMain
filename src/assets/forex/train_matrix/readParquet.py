import pandas as pd

df  = pd.read_parquet('barrier_check.parquet')

df = df[:5000]
df.to_csv('barrier_check.csv')