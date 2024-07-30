import pandas as pd
import numpy as np
import time
from pandarallel import pandarallel
import time
pandarallel.initialize()

def calculate_barriers_R(df, lookback):
    start_time = time.time()
    # Calculate volatility
    df['pct_change'] = df['Close'].pct_change()
    volatility = df['pct_change'].rolling(window=lookback).std()
    
    # Add necessary columns
    df['Datetime'] = pd.to_datetime(df['Date'])
    df['unix'] = df['Datetime'].astype('int64') // 10**9
    df['upper_barrier'] = df['Close'] * (1 + 1 * volatility)
    df['lower_barrier'] = df['Close'] * (1 - 1 * volatility)
    
    # Calculate lookback in seconds
    lookback_seconds = lookback * 3600
    df['endbarrier_unix'] = df['unix'] + lookback_seconds

    # Copy df to price_df and keep only 'unix' and 'Close'
    price_df = df[['unix', 'Close']].copy()

    # Convert price_df to a numpy array
    price_df_values = price_df.values

    # Function to find prices between unix and endbarrier_unix
    def find_prices_in_range(start_unix, end_unix):
        mask = (price_df_values[:, 0] >= start_unix) & (price_df_values[:, 0] <= end_unix)
        return price_df_values[mask, 1]

    # Apply the function to each row
    df['prices_in_range'] = df.parallel_apply(lambda row: find_prices_in_range(row['unix'], row['endbarrier_unix']), axis=1)

    # Function to check if prices hit barriers and which hits first
    def price_barrier_check(row):
        lower_barrier = row['lower_barrier']
        upper_barrier = row['upper_barrier']
        prices = row['prices_in_range']
        
        # Use numpy to find the first occurrence of crossing the barriers
        if prices.size == 0:
            return 0, np.nan
        
        upper_hits = np.where(prices > upper_barrier)[0]
        lower_hits = np.where(prices < lower_barrier)[0]
        
        if upper_hits.size == 0 and lower_hits.size == 0:
            return 0, np.nan
        
        if upper_hits.size > 0 and (lower_hits.size == 0 or upper_hits[0] < lower_hits[0]):
            return 1, prices[upper_hits[0]]
        elif lower_hits.size > 0 and (upper_hits.size == 0 or lower_hits[0] < upper_hits[0]):
            return -1, prices[lower_hits[0]]

    # Apply the price_barrier_check function to each row
    df[['label', 'touch_price']] = df.parallel_apply(price_barrier_check, axis=1, result_type='expand')
    
    #df.to_csv('updated_df.csv')
    end_time = time.time()

    runtime =end_time - start_time
    print(f'function runtime: {runtime}')
    return df

# Example usage

period =1000000
data = {
    'Date': pd.date_range(start='2022-01-01', periods=period, freq='h'),
    'Close': np.random.rand(period) * period
}
df = pd.DataFrame(data)
lookback = 8  # 24 hours lookback period
result_df = calculate_barriers_R(df, lookback)
#result_df.to_csv('barrier_pricecheck.csv')
print(result_df)
