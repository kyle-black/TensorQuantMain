from tvDatafeed import TvDatafeed, Interval
import pandas as pd

def latest_data(securities):
    username = 'StoCASHtic-ML'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)

    df_list = []

    for i in securities:
        nifty_index_data = tv.get_hist(symbol=f'{i}', exchange='OANDA', interval=Interval.in_1_minute, n_bars=10000)

        # Reset index to keep the datetime as a column for merging
        df = nifty_index_data.reset_index()

        # Rename columns to make them unique for each security
        df.rename(columns={'datetime': 'Date', 
                           'open': f'{i}_Open', 
                           'high': f'{i}_High', 
                           'low': f'{i}_Low', 
                           'close': f'{i}_Close',
                           'volume': f'{i}_Volume'}, inplace=True)

        # Set 'Date' as the index for this DataFrame
        df.set_index('Date', inplace=True)

        df_list.append(df)

    # Concatenate all DataFrames along the 'Date' index
    df_all = pd.concat(df_list, axis=1)

    return df_all

if __name__ == "__main__":
    security_list = ['EURUSD', 'AUDUSD', 'USDCAD', 'USDCHF']
    combined_df = latest_data(security_list)
    print(combined_df)
