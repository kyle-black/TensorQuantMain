from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import datetime

def latest_data_5(security):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

   
    
    

    nifty_index_data = tv.get_hist(symbol=f'{security}',exchange='OANDA',interval=Interval.in_5_minutes,n_bars=1000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

   
    return df


def latest_data_15(security):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

  
    
    

    nifty_index_data = tv.get_hist(symbol=f'{security}',exchange='OANDA',interval=Interval.in_15_minutes,n_bars=1000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

    
    return df


def latest_data_30(security):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

   
    nifty_index_data = tv.get_hist(symbol=f'{security}',exchange='OANDA',interval=Interval.in_30_minutes,n_bars=1000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

   
    return df

def latest_data_60(security):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

    nifty_index_data = tv.get_hist(symbol=security ,exchange='OANDA',interval=Interval.in_1_hour,n_bars=1000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

    return df
