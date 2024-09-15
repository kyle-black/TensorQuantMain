import pandas as pd
import numpy as np
pd.set_option('display.float_format', '{:.2f}'.format)
# Define the function to simulate trades
def trade_dataframe_creator(df):
    # Filter the trades based on the probability classes
    #df = df.query('Proba_Class_0 > 0.62 or Proba_Class_1 > 0.62')

    

    # Select the randomized trades
   # selected_trades = df.iloc[random_trade].copy()
    selected_trades =df.copy()
    # Calculate the percentage change
    selected_trades['pct_change'] = selected_trades['Close'] / selected_trades['touch_price']

    # Initialize columns
    selected_trades['Choice'] = np.nan
    selected_trades['Accurate'] = np.nan
    selected_trades['major_proba'] = np.nan

    # Set 'Choice' and 'major_proba' based on probabilities
    for idx, row in selected_trades.iterrows():
        if row['Proba_Class_0'] >= 0.50:
            selected_trades.at[idx, 'Choice'] = 0
            selected_trades.at[idx, 'major_proba'] = row['Proba_Class_0']
        elif row['Proba_Class_1'] >= 0.50:
            selected_trades.at[idx, 'Choice'] = 1
            selected_trades.at[idx, 'major_proba'] = row['Proba_Class_1']

    # Set 'Accurate' based on whether the choice matches the true label
    selected_trades['Accurate'] = selected_trades['Choice'] == selected_trades['True_Label']
    selected_trades['scaled_proba'] = (selected_trades['major_proba'] - selected_trades['major_proba'].min()) / (selected_trades['major_proba'].max() - selected_trades['major_proba'].min())
    selected_trades = selected_trades.query('Proba_Class_0 > 0.56 or Proba_Class_1 > 0.56')
    
    # Randomize Trading sequence50
    random_trade = np.random.choice(len(selected_trades), len(selected_trades), replace=False)
    random_trade = np.sort(random_trade)
    selected_trades = selected_trades.iloc[random_trade].copy()

    return selected_trades


# Calculate trades and adjust based on the account balance
def trade_calculate(df, leverage, base_lot_size):
    # Apply min-max scaling to probabilities
    

    # Calculate adjusted lot size based on the scaled probability and base lot size
    df['adjusted_lot_size'] = df['scaled_proba'] * base_lot_size
    pip_movement = (df['touch_price'] - df['Close'])/ 0.0001
    # Calculate pip movement (difference between touch_price and Close)
    df['abs_pip_movement'] = abs(pip_movement)  # 0.0001 represents 1 pip in forex
    #df['net_pip_movement'] = df['touch_price'] - df['Close']/ 0.0001

    df['net_pip_movement'] = np.where(df['Accurate'], pip_movement, -pip_movement)
    # Calculate the pip value based on the adjusted lot size and Close price
    df['pip_value'] = (0.0001 * df['adjusted_lot_size']) / df['Close']
    
    # Calculate profit or loss in real dollars (pip movement multiplied by pip value)
    df['net_dollar_value'] = df['abs_pip_movement'] * df['pip_value']
    
    # Calculate Profit/Loss based on whether the trade was accurate
    df['Profit_Loss'] = np.where(df['Accurate'], df['net_dollar_value'], -df['net_dollar_value'])

    return df


# Simulate trades and adjust account balance and lot size
def trade_append(df, initial_balance, leverage, base_lot_size):
    # Initialize account balance column
    df.reset_index(inplace=True)
    df['account_balance'] = initial_balance

    # Loop through each row, calculate the new balance and adjust lot size
    for i in range(len(df)):
        if i == 0:
            # First row, initialize account balance and calculate profit/loss
            df.at[i, 'account_balance'] = initial_balance + df.at[i, 'Profit_Loss']
        else:
            # Adjust the account balance based on previous balance and current Profit/Loss
            df.at[i, 'account_balance'] = df.at[i - 1, 'account_balance'] + df.at[i, 'Profit_Loss']

        # Adjust the base lot size based on the curre
        # nt account balance
        current_balance = df.at[i, 'account_balance']

        print('Current Balance:', current_balance)
        #adjusted_lot_size = (current_balance / initial_balance) * base_lot_size

        adjusted_lot_size = base_lot_size 
        # Recalculate adjusted lot size based on the updated account balance
        df = trade_calculate(df, leverage, adjusted_lot_size)
    
    df = df.round({'Profit_Loss': 2, 'account_balance': 0})


    return df


# Main logic
if __name__ == "__main__":
    df = pd.read_csv('testfiles/test_result_EURUSD_0816-364.csv')

    # Set initial parameters
    initial_balance = 2000
    leverage = 50
    base_lot_size = 10000

    # Step 1: Create the trade DataFrame
    new_df = trade_dataframe_creator(df)

    # Step 2: Calculate the trades based on initial lot size
    new_df = trade_calculate(new_df, leverage, base_lot_size)

    # Step 3: Append account balance and adjust lot size based on account growth or contraction
    new_df = trade_append(new_df, initial_balance, leverage, base_lot_size)

    # Save the result
    new_df.to_csv('traded_df22.csv')

    print(new_df)