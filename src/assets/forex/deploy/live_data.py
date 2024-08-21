from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import datetime
import requests
import json


 # Make sure you have the required import for TvDatafeed
'''
def latest_data(securities):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

    df_list = []

    for i in securities:
        nifty_index_data = tv.get_hist(symbol=f'{i}', exchange='OANDA', interval=Interval.in_1_minute, n_bars=50000)

        df = nifty_index_data
        df = df.reset_index()
        df.rename(columns={'datetime': f'{i}_Date', 'open': f'{i}_Open', 'high': f'{i}_High', 'low': f'{i}_Low', 'close': f'{i}_Close', 'volume': f'{i}_Volume'}, inplace=True)

        df_list.append(df)
    
   # print(df_list)
    # Combine the data for all securities along the columns
    df_all = pd.concat(df_list, axis=1)
  #  df_all = df_all.dropna()

    df_all = df_all.set_index('EURUSD_Date')

    
    return df_all

'''



def latest_data(symbol):
    url ='https://financialmodelingprep.com/api/v3/historical-chart/1min/EURUSD?from=2024-08-15&to=2024-08-20&apikey=3e17d2b777a13feee4c1243985cdc7c4'

    response = requests.get(url)

       
    print(response.json())
    data = response.json()


    df =pd.DataFrame(data)

    print('length of data:', df)            
 
    

        
        
            
    with open(f'{symbol}.json', 'a') as f:
        json.dump(data, f)

if __name__ == "__main__":
    #security_list = ['EURUSD','AUDUSD']
    
    
    df = latest_data('EURUSD')
    print(df)