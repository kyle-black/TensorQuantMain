import pandas as pd


df =pd.read_csv('merged.csv')

df =df.tail(1000)
df.to_csv('tester.csv')
