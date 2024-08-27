import pandas as pd
import numpy as np
from scipy.stats import boxcox





def dollar_bar_creator(asset, df_, dollar_amt):
    # Convert the date column to datetime format to avoid sorting issues
    df_['EURUSD_Date'] = pd.to_datetime(df_['EURUSD_Date'], errors='coerce')
    
    # Drop rows where the conversion failed (i.e., rows with invalid dates)
    df_.dropna(subset=['EURUSD_Date'], inplace=True)

    # Sort the DataFrame by date
    df_.sort_values('EURUSD_Date', inplace=True)

    close = f'{asset}_Close'
    volume = f'{asset}_Volume'

    vol_count = 0
    dollar_count = 0
    new_bar = []

    for i, d in df_.iterrows():
        dollar_count += round((d[volume] * d[close]), 2)

        if dollar_count >= dollar_amt:
            bar = {'Date': i, 'Close': d[close], 'Volume': d[volume], 'AUDUSD_Close': d['AUDUSD_Close']}
            new_bar.append(bar)
            dollar_count = 0

    new_bar_df = pd.DataFrame(new_bar)

    new_bar_df['pips'] = new_bar_df['Close'] * 10000
    new_bar_df['change'] = new_bar_df['pips'].diff()
    new_bar_df['pct_change'] = new_bar_df['change'].pct_change()

    new_bar_df['Datehold'] = pd.to_datetime(new_bar_df['Date'])
    new_bar_df['day_of_week'] = new_bar_df['Datehold'].dt.dayofweek

    return new_bar_df


def security_append(main_df, security_list):
    for asset_ in security_list:
        df_ = pd.read_csv(f'updated_data/{asset_}_1.csv')
        df_.sort_values('Date', inplace=True)
        df_ = df_.set_index('Date')
        df_ = df_[['Close', 'Volume']]

        # Join the dataframe with the main dataframe
        main_df = main_df.join(df_, how='left', rsuffix=f'_{asset_}')

    return main_df

    
    






    #return new_bar

'''
if __name__ in "__main__":

    df = pd.read_csv('updated_data/EURUSD.csv')

    print(df.head())

    print(dollar_bar_creator(df))
'''