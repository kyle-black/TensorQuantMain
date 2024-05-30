import pandas as pd





def dollar_bar_creator(asset,df_,dollar_amt):

    #df_ = pd.read_csv('merged.csv')
    df_.sort_values('Date', inplace =True)
    
    close = f'Close_{asset}'
    volume = f'Volume_{asset}'
    
    df_ = df_.set_index('Date')


   # df_ = df_[[close,volume]]

    #df_ = df_.tail(500000)


    vol_count = 0
    dollar_count = 0
    new_bar = []

    for i, d in df_.iterrows():


        dollar_count+=round((d[volume] * d[close]),2)
        #vol_count += d[volume]

        if dollar_count >= dollar_amt:
            bar ={'Date':i,'Close':d[close],'Volume':d[volume],'Close_AUDUSD':d['Close_AUDUSD'],'Close_USDCAD':d['Close_USDCAD'],'Close_USDCHF':d['Close_USDCHF'],'AUDUSD_Returns':d['AUDUSD_Returns'],'USDCAD_Returns':d['USDCAD_Returns'],'USDCHF_Returns':d['USDCHF_Returns'], 'durableGoods':d['durableGoods'], '15Yr_Fixed':d['durableGoods'], '30Yr_Fixed':d['30Yr_Fixed'], 'CPI':d['CPI'],
       'GDP':d['GDP'], 'Production_Total_Index':d['Production_Total_Index'], 'Yields_COD':d['Yields_COD'], 'consumerSentiment':d['consumerSentiment'],
       'federalFunds':d['federalFunds'], 'inflation':d['inflation'], 'inflationRate':d['inflationRate'], 'initialClaims':d['initialClaims'],
       'nominalPotentialGDP':d['nominalPotentialGDP'], 'rates_CreditCards':d['rates_CreditCards'], 'realGDP':d['realGDP'],
       'realGDPPerCapita':d['realGDPPerCapita'], 'retailMoneyFunds':d['retailMoneyFunds'], 'retailSales':d['retailSales']}
            new_bar.append(bar)
         #   vol_count = 0 
            dollar_count = 0
         
        


        
        #print(dollar_count)
        #print(i,d)

    new_bar_df = pd.DataFrame(new_bar)
    new_bar_df['pct_change'] = new_bar_df['Close'].pct_change()
    pct_change_mean = new_bar_df['pct_change'].mean()
    pct_change_std = new_bar_df['pct_change'].std()

    new_bar_df['normal_pct_change'] = ((new_bar_df['pct_change']-pct_change_mean) / pct_change_std)


    #new_bar_df['Returns_100'] = new_bar_df['Returns'] *100


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