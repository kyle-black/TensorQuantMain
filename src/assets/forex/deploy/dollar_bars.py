import pandas as pd
import numpy as np
from scipy.stats import boxcox





def dollar_bar_creator(asset,df_,dollar_amt):

    df_.rename(columns={'date':'Date','open':'Open','low':'Low','high':'High','close':'Close','volume':'Volume'}, inplace=True)
    print('dollar bar df:', df_)
    #df_ = pd.read_csv('merged.csv')
    df_.sort_values('Date', inplace =True)
    #df_.sort_index()

    df_ = df_[:]
    
    close = f'{asset}_Close'
    volume = f'{asset}_Volume'
    
   # df_ = df_.set_index('Date')

    #for i in ['AUDUSD','USDCAD','USDCHF']:
    #    df_[f'{i}_Returns'] = df_[f'Close_{asset}'].pct_change()


   # df_ = df_[[close,volume]]

    #df_ = df_.tail(500000)


    vol_count = 0
    dollar_count = 0
    new_bar = []


    for i, d in df_.iterrows():


        dollar_count+=round((d[volume] * d[close]),2)
        #vol_count += d[volume]

        if dollar_count >= dollar_amt:
            bar ={'Date':i,'Close':d[close],'Volume':d[volume],'AUDUSD_Close':d['AUDUSD_Close']}
            new_bar.append(bar)
         #   vol_count = 0 
            dollar_count = 0
         
        


        
        #print(dollar_count)
        #print(i,d)

    new_bar_df = pd.DataFrame(new_bar)

    new_bar_df['pips'] = new_bar_df['Close'] *10000
    new_bar_df['change'] = (new_bar_df['pips'].diff()) 
    new_bar_df['pct_change'] = new_bar_df['change'].pct_change()

    new_bar_df['Datehold'] = pd.to_datetime(new_bar_df['Date'])
    new_bar_df['day_of_week'] = new_bar_df['Datehold'].dt.dayofweek
   # new_bar_df.drop('Datehold', inplace=True)

   


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