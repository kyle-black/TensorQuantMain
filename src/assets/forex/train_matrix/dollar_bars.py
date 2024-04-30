import pandas as pd





def dollar_bar_creator(asset,dollar_amt):

    df_ = pd.read_csv(f'updated_data/{asset}_1.csv')
    df_.sort_values('Date', inplace =True)
    

    
    df_ = df_.set_index('Date')


    df_ = df_[['Close','Volume']]

    #df_ = df_.tail(500000)


    vol_count = 0
    dollar_count = 0
    new_bar = []

    for i, d in df_.iterrows():


        dollar_count+=round((d['Volume'] * d['Close']),2)


        if dollar_count >= dollar_amt:
            bar ={'Date':i,'Close':d['Close']}
            new_bar.append(bar)
            dollar_count = 0
         
        


        
        #print(dollar_count)
        #print(i,d)

    new_bar_df = pd.DataFrame(new_bar)
    new_bar_df['Returns'] = new_bar_df['Close'].pct_change()
    new_bar_df['Returns_100'] = new_bar_df['Returns'] *100


    return new_bar_df





    #return new_bar

'''
if __name__ in "__main__":

    df = pd.read_csv('updated_data/EURUSD.csv')

    print(df.head())

    print(dollar_bar_creator(df))
'''