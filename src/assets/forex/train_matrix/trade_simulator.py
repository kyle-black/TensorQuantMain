import pandas as pd
import numpy as np

df = pd.read_csv('tester_df.csv')

print(df.head())

account = 10000
random_trade = np.random.randint(4000, size=(100))

random_trade =np.sort(random_trade)

# Initialize an empty list to store the selected trades
selected_trades = []

def make_trade(account, ledger, randomizer):
    ### create trade randomizer
    for i in random_trade:
        trade = df.iloc[i]
        selected_trades.append(trade)
        print(trade)

# Call the function
make_trade(account, None, random_trade)

# Create a new DataFrame from the selected trades
new_df = pd.DataFrame(selected_trades)

# Show the new DataFrame
print(new_df.head())