from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import datetime
import json
import schedule
import time
#from redis_connect import url_connection


import redis
#from urllib.parse import urlparse  



REDIS_URL = "redis://default:zHeoOL4uqpzaxTC7YgtuWvq4HRNSsoD0@redis-17905.c326.us-east-1-3.ec2.cloud.redislabs.com:17905"

# Create a Redis connection
url_connection = redis.from_url(REDIS_URL)




def latest_data_5(security):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

   
    
    

    nifty_index_data = tv.get_hist(symbol=f'{security}',exchange='OANDA',interval=Interval.in_5_minute,n_bars=1000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

    json_str = df.to_json(orient='split')
    url_connection.set(f'{security}_5m', json_str)

   
    return df


def latest_data_15(security):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

  
    
    

    nifty_index_data = tv.get_hist(symbol=f'{security}',exchange='OANDA',interval=Interval.in_15_minute,n_bars=1000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

    json_str = df.to_json(orient='split')
    url_connection.set(f'{security}_15m', json_str)
    
    return df


def latest_data_30(security):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

   
    nifty_index_data = tv.get_hist(symbol=f'{security}',exchange='OANDA',interval=Interval.in_30_minute,n_bars=1000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

    json_str = df.to_json(orient='split')
    url_connection.set(f'{security}_30m', json_str)
   
    return df

def latest_data_60(security, window_length):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

    nifty_index_data = tv.get_hist(symbol=f'{security}',exchange='OANDA',interval=Interval.in_1_hour,n_bars=1000)

    df = nifty_index_data

    df = df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

    # Get the last date in the DataFrame
    last_date = df['Date'].iloc[-1]

    # Add 48 hours to the last date
    new_date = last_date + pd.Timedelta(hours=window_length)

    # Add the new date to the DataFrame
    
    df['EndDate'] = new_date
    #df = df.append({'EndDate': new_date}, ignore_index=True)

    json_str = df.to_json(orient='split')
    
    url_connection.set(f'{security}_60m', json_str)

    return df



if __name__ == "__main__":
    window_length = 48
    def job_60():
        latest_data_60('EURUSD', window_length )
        latest_data_60('AUDUSD', window_length)

# Schedule the job every hour
    schedule.every().hour.at(":00").do(job_60)

# Main loop
    while True:
    # Run pending scheduled jobs
        schedule.run_pending()
        time.sleep(1)


    '''

    latest_data_5('EURUSD')
    latest_data_15('EURUSD')
    latest_data_30('EURUSD')
    latest_data_60('EURUSD')
    latest_data_5('AUDUSD')
    latest_data_15('AUDUSD')
    latest_data_30('AUDUSD')
    latest_data_60('AUDUSD')
    latest_data_5('GBPUSD')
    latest_data_15('GBPUSD')
    latest_data_30('GBPUSD')
    latest_data_60('GBPUSD')
    latest_data_5('USDJPY')
    latest_data_15('USDJPY')
    latest_data_30('USDJPY')
    latest_data_60('USDJPY')
    latest_data_5('USDCHF')
    latest_data_15('USDCHF')
    latest_data_30('USDCHF')
    latest_data_60('USDCHF')
    latest_data_5('USDCAD')
    latest_data_15('USDCAD')
    latest_data_30('USDCAD')
    latest_data_60('USDCAD')
    '''
    
    