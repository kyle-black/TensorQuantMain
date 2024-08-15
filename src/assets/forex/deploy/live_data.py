from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import datetime




def latest_data(securities):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

   # securities_list = ['AUDUSD','EURGBP','EURUSD', 'USDCAD', 'USDCHF','USDCNH','USDHKD','USDJPY']
    #df_all = pd.DataFrame()

    df_list =[]
    #for security in securities_list:
    
    #security = 'EURUSD'

    for i in securities:
        nifty_index_data = tv.get_hist(symbol=f'{i}',exchange='OANDA',interval=Interval.in_1_minute,n_bars=10000)

        df= nifty_index_data

        df =df.reset_index()
        df.rename(columns = {'datetime':f'{i}_Date','open': f'{i}_Open', 'high':f'{i}_High', 'low':f'{i}_Low', 'close':f'{i}_Close','volume':f'{i}_Volume'}, inplace=True)

        df_list.append(df)
    
    df_all = pd.concat(df_list, ignore_index =True)

    df.set_index('EURUSD_Date', inplace=True)
    return df



if __name__ in "__main__":
    security_list = 'EURUSD','AUDUSD', 'CADUSD', 'USDCHF'
    latest_data(security_list)
    print(latest_data)
