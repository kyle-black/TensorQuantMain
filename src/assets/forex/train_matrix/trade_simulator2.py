import pandas as pd
import numpy as np

# Define the function to simulate trades
def trade_dataframe_creator(df):
    df = df.query('Proba_Class_0 > 0.50 or Proba_Class_1 > 0.50')
    
    data_length = len(df)
    print('data_length', data_length)

    random_trade = np.random.choice(data_length, size=600, replace=False)
    random_trade = np.sort(random_trade)

    selected_trades = []
    for i in random_trade:
        trade = df.iloc[i]
        selected_trades.append(trade)
        print(i)

    new_df = pd.DataFrame(selected_trades)

    new_df['pct_change'] = new_df['Close'] / new_df['touch_price']

    new_df['Choice'] = np.nan
    new_df['Accurate'] = np.nan
    new_df['major_proba'] = np.nan
    new_df['Accurate'] = new_df['Accurate'].astype(bool)

    for idx, row in new_df.iterrows():
        if row['Proba_Class_0'] >= 0.50:
            new_df.at[idx, 'Choice'] = 0
            new_df.at[idx, 'major_proba'] = row['Proba_Class_0']
        elif row['Proba_Class_1'] >= 0.50:
            new_df.at[idx, 'Choice'] = 1
            new_df.at[idx, 'major_proba'] = row['Proba_Class_1']

    for idx, row in new_df.iterrows():
        if row['Choice'] == row['True_Label']:
            new_df.at[idx, 'Accurate'] = True
        else:
            new_df.at[idx, 'Accurate'] = False

    new_df = new_df[['Close', 'touch_price', 'upper_barrier', 'lower_barrier', 'pct_change', 'Proba_Class_0', 'Proba_Class_1', 'True_Label', 'Choice', 'Accurate', 'major_proba']]

    return new_df


def trade_calculate(df, leverage, account, base_lot_size, epsilon=1e-6):
    df['log_scaled_proba'] = np.log(df['major_proba'] + epsilon)
    df['log_scaled_proba'] = df['log_scaled_proba'] / df['log_scaled_proba'].max()

    for idx, row in df.iterrows():
        # Adjust the lot size to be 10x the current account balance
        adjusted_lot_size = 10 * account
        df.at[idx, 'adjusted_lot_size'] = adjusted_lot_size

        # Apply log scaling to adjust lot size
        df.at[idx, 'log_scaled_lot_size'] = adjusted_lot_size * row['log_scaled_proba']

        total_trade_size = df.at[idx, 'log_scaled_lot_size'] * row['Close']
        required_margin = total_trade_size / leverage

        df.at[idx, 'start_pip_value'] = 0.0001 * df.at[idx, 'log_scaled_lot_size'] / row['Close']
        df.at[idx, 'end_pip_value'] = 0.0001 * df.at[idx, 'log_scaled_lot_size'] / row['touch_price']

        net_pips = df.at[idx, 'end_pip_value'] - df.at[idx, 'start_pip_value']
        net_trade_value = net_pips * df.at[idx, 'log_scaled_lot_size']
        df.at[idx, 'net_trade_value'] = net_trade_value

        if row['Accurate']:
            profit_loss = abs(net_trade_value) if net_trade_value > 0 else net_trade_value
        else:
            profit_loss = -abs(net_trade_value) if net_trade_value > 0 else net_trade_value

        df.at[idx, 'Profit_Loss'] = profit_loss

        # Update the account balance after each trade
        account += profit_loss
        print(f'Updated account balance after trade {idx}: {account}')

    return df, account


def trade_simulate(df, account, leverage, base_lot_size):
    new_df = trade_dataframe_creator(df)
    new_df, account = trade_calculate(new_df, leverage, account, base_lot_size)
    print(f'Final account balance: {account}')
    return account


if __name__ == "__main__":
    df = pd.read_csv('testfiles/test_result_EURUSD_0816-9.csv')
    account = 1000
    leverage = 50
    base_lot_size = 10000

    final_account_balance = trade_simulate(df, account, leverage, base_lot_size)
    print(f'Final Account Balance: {final_account_balance}')
   # new_df.to_csv('traded_df20.csv')

