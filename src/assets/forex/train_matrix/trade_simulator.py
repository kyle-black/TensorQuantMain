import pandas as pd
import numpy as np


# Define the function to simulate trades
def trade_dataframe_creator(df):

    df = df.query('Proba_Class_0 > 0.50 or Proba_Class_1 > 0.50')
    
    data_length = len(df)
    print('data_length',data_length)
    #Randomize Trading sequence
   # random_trade = np.sort(np.random.randint(data_length, size=(600)))

    random_trade = np.random.choice(data_length, size=100, replace=False)
    random_trade = np.sort(random_trade)
   # random_trade = np.random.randint(data_length, size=(100))
    selected_trades = []
    for i in random_trade:
        trade = df.iloc[i]
        selected_trades.append(trade)
        print(i)

    # Create a new DataFrame from the selected trades
    new_df = pd.DataFrame(selected_trades)

    # Calculate the percentage change
    new_df['pct_change'] = new_df['Close'] / new_df['touch_price']

    # Initialize the 'Choice' column
    new_df['Choice'] = np.nan
    new_df['Accurate'] = np.nan
    new_df['major_proba'] = np.nan
    new_df['Accurate'] = new_df['Accurate'].astype(bool)
    # Iterate over the DataFrame and set 'Choice' based on the probabilities
    for idx, row in new_df.iterrows():
        if row['Proba_Class_0'] >= 0.50:
            new_df.at[idx, 'Choice'] = 0
            new_df.at[idx,'major_proba'] = row['Proba_Class_0']
        elif row['Proba_Class_1'] >= 0.50:
            new_df.at[idx, 'Choice'] = 1
            new_df.at[idx,'major_proba'] = row['Proba_Class_1']

    for idx, row in new_df.iterrows():
        if row['Choice'] == row['True_Label']:
            new_df.at[idx, 'Accurate'] = True
        else:   new_df.at[idx, 'Accurate'] = False

    new_df = new_df[['Close','touch_price', 'upper_barrier', 'lower_barrier', 'pct_change', 'Proba_Class_0', 'Proba_Class_1', 'True_Label', 'Choice', 'Accurate', 'major_proba']]

    return new_df



def trade_calculate(df, leverage, base_lot_size, epsilon=1e-6):

    # Apply logarithmic scaling to the probability
    df['log_scaled_proba'] = np.log(df['major_proba'] + epsilon)

    # Normalize the log-scaled probabilities to a reasonable range
    df['log_scaled_proba'] = df['log_scaled_proba'] / df['log_scaled_proba'].max()

    # Calculate adjusted lot size based on the log-scaled probability
    df['adjusted_lot_size'] = df['log_scaled_proba'] * base_lot_size

    # Calculate total trade size (e.g., 10000 * 1.07672)
    total_trade_size = df['adjusted_lot_size'] * df['Close']

    # Calculate required margin (e.g., (10,767.20 / 50) for 50:1 leverage)
    required_margin = total_trade_size / leverage

    # Calculate starting pip value
    df['start_pip_value'] = 0.0001 * df['adjusted_lot_size'] / df['Close']

    # Calculate end pip value (based on touch price)
    df['end_pip_value'] = 0.0001 * df['adjusted_lot_size'] / df['touch_price']

    # Initialize 'net_pips' column
    df['net_pip_value'] = np.nan
    df['net_trade_value'] = np.nan
    df['Profit_Loss'] = np.nan

    # Iterate through each row to calculate 'net_pips'
    for idx, row in df.iterrows():
        
        net_pips = row['end_pip_value'] - row['start_pip_value']
        df.at[idx, 'net_pip_value'] = net_pips

        net_trade_value = net_pips * row['adjusted_lot_size']
        df.at[idx, 'net_trade_value'] = net_trade_value
        
        if row['Accurate']:
            if net_trade_value > 0:
                profit_loss = net_trade_value
            else:
                profit_loss = abs(net_trade_value)
        else:
            if net_trade_value > 0:
                profit_loss = -net_trade_value
            else:
                profit_loss = net_trade_value
                
        df.at[idx, 'Profit_Loss'] = profit_loss

    return df


   
def trade_simulate(df,account):

    for idx, row in df.iterrows():
        account += row['Profit_Loss']

        print('current_account balance',account) 
    return account



#if __name__ in "_main__":

df = pd.read_csv('tester_df6.csv')

# Set initial account balance
account = 10000

### Simple
new_df = trade_dataframe_creator(df)
print(new_df)
leverage =50
lot_size =10000

new_df = trade_calculate(new_df, leverage, lot_size)
print(new_df)
print(trade_simulate(new_df, account))

new_df.to_csv('traded_df.csv')




        


