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
    dflist = []

    # EURUSD
    url = 'https://financialmodelingprep.com/api/v3/historical-chart/1min/EURUSD?from=2024-08-27&to=2024-08-27&apikey=3e17d2b777a13feee4c1243985cdc7c4'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df.rename(columns={'date': 'EURUSD_Date', 'open': 'EURUSD_Open', 'high': 'EURUSD_High', 'low': 'EURUSD_Low', 'close': 'EURUSD_Close', 'volume': 'EURUSD_Volume'}, inplace=True)
    dflist.append(df)

    # AUDUSD
    aud_url = 'https://financialmodelingprep.com/api/v3/historical-chart/1min/AUDUSD?from=2024-08-15&to=2024-08-21&apikey=3e17d2b777a13feee4c1243985cdc7c4'
    aud_response = requests.get(aud_url)
    aud_data = aud_response.json()
    aud_df = pd.DataFrame(aud_data)
    aud_df.rename(columns={'date': 'AUDUSD_Date', 'open': 'AUDUSD_Open', 'high': 'AUDUSD_High', 'low': 'AUDUSD_Low', 'close': 'AUDUSD_Close', 'volume': 'AUDUSD_Volume'}, inplace=True)
    dflist.append(aud_df)

    # Combine dataframes
    df_all = pd.concat(dflist, axis=1)
    df_all.set_index('EURUSD_Date', inplace=True)

    # Writing DataFrame to JSON file
    with open(f'updated_data/{symbol}_live.json', 'a') as f:
        json.dump(df_all.to_dict(orient='records'), f)

    return df_all


def combine_data():
    
    dflist = []
    eur_df = pd.read_json('updated_data/EURUSD.json')

    eur_df.rename(columns={'date': f'EURUSD_Date', 'open': f'EURUSD_Open', 'high': f'EURUSD_High', 'low': f'EURUSD_Low', 'close': f'EURUSD_Close', 'volume': f'EURUSD_Volume'}, inplace=True)
    dflist.append(eur_df)


    aud_df = pd.read_json('updated_data/AUDUSD.json')
    aud_df.rename(columns={'date': f'AUDUSD_Date', 'open': f'AUDUSD_Open', 'high': f'AUDUSD_High', 'low': f'AUDUSD_Low', 'close': f'AUDUSD_Close', 'volume': f'AUDUSD_Volume'}, inplace=True)
    dflist.append(aud_df)

    
    #live_df = pd.read_json('updated_data/EURUSD_live.json')
    
   # dflist.append(live_df)


    df_all = pd.concat(dflist, axis=1)

    df_all = df_all.set_index('EURUSD_Date')
    df_all = df_all.dropna()

    df_all.to_csv('updated_data/combined_df.csv')

    return df_all







if __name__ == "__main__":
    #security_list = ['EURUSD','AUDUSD']
    
    
    df = combine_data()
    print(df)
