from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import datetime




def latest_data_60():
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

   # securities_list = ['AUDUSD','EURGBP','EURUSD', 'USDCAD', 'USDCHF','USDCNH','USDHKD','USDJPY']
    #df_all = pd.DataFrame()

    #df_list =[]
    #for security in securities_list:
    
    security = 'EURUSD'

    nifty_index_data = tv.get_hist(symbol=f'{security}',exchange='OANDA',interval=Interval.in_1_minute,n_bars=10000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

    #df_list.append(df)
    
    #df_all = pd.concat(df_list, ignore_index =True)

   # df.set_index('Date', inplace=True)
    return df



if __name__ in "__main__":
    print(latest_data_60())
