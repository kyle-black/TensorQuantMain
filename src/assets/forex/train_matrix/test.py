import pandas as pd


df =pd.read_csv('merged.csv')

df =df.tail(100)
df.to_csv('tester.csv')
