import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt





df = pd.read_csv('merged.csv')


print(df.columns)


df =df[['Close_EURUSD', 'Volume_EURUSD','Close_USDCHF', 'Close_USDCAD', 'Close_USDHKD', 'Close_AUDUSD',
       'Close_USDJPY', 'durableGoods', '15Yr_Fixed', '30Yr_Fixed', 'CPI',
       'GDP', 'Production_Total_Index', 'Yields_COD', 'consumerSentiment',
       'federalFunds', 'inflation', 'inflationRate', 'initialClaims',
       'nominalPotentialGDP', 'rates_CreditCards', 'realGDP',
       'realGDPPerCapita', 'retailMoneyFunds', 'retailSales']]

#df.drop(columns=['Date'])


correlation_matrix =df.corr()

print(correlation_matrix)



plt.figure(figsize=(20, 20))  # Optional: You can set the figure size
sns.heatmap(correlation_matrix, annot=True)

plt.savefig('correlation_matrix.png')  # Save the figure

