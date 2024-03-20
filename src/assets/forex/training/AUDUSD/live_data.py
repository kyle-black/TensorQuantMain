from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import datetime
import json
from redis_connect import url_connection

def latest_data_5(security):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

   
    
    

    nifty_index_data = tv.get_hist(symbol=f'{security}',exchange='OANDA',interval=Interval.in_5_minutes,n_bars=10000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

   
    return df


def latest_data_15(security):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

  
    
    

    nifty_index_data = tv.get_hist(symbol=f'{security}',exchange='OANDA',interval=Interval.in_15_minutes,n_bars=10000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

    
    return df


def latest_data_30(security):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

   
    nifty_index_data = tv.get_hist(symbol=f'{security}',exchange='OANDA',interval=Interval.in_30_minutes,n_bars=10000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

   
    return df

def latest_data_60(security):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

    nifty_index_data = tv.get_hist(symbol=f'{security}',exchange='OANDA',interval=Interval.in_1_hour,n_bars=10000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

    json_str = df.to_json(orient='split')
    
    url_connection.set(f'{security}_60m', json_str)


    return df
