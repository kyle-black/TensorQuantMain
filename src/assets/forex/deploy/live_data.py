from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import datetime


 # Make sure you have the required import for TvDatafeed

def latest_data(securities):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

    df_list = []

    for i in securities:
        nifty_index_data = tv.get_hist(symbol=f'{i}', exchange='OANDA', interval=Interval.in_1_minute, n_bars=10000)

        df = nifty_index_data
        df = df.reset_index()
        df.rename(columns={'datetime': f'{i}_Date', 'open': f'{i}_Open', 'high': f'{i}_High', 'low': f'{i}_Low', 'close': f'{i}_Close', 'volume': f'{i}_Volume'}, inplace=True)

        df_list.append(df)
    
    # Combine the data for all securities along the columns
    df_all = pd.concat(df_list, axis=1)
    
    return df_all

if __name__ == "__main__":
    security_list = ['AUDUSD']
    df = latest_data(security_list)
    print(df)