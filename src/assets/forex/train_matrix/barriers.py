import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay, DateOffset, WeekOfMonth, LastWeekOfMonth
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import Day, BMonthEnd
import time
from pandarallel import pandarallel
pandarallel.initialize()




def calculate_barriers_R(df, lookback):
    start_time = time.time()
    #new_bar_df['pips'] = new_bar_df['Close'] *10000
    
    # Ensure 'Date' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Set 'Date' as index for resampling
    df.set_index('Date', inplace=True)
    
    df['pct_change'] = df['Close'].pct_change()

# Calculate rolling standard deviation (volatility)
    df['volatility'] = df['pct_change'].rolling(window=lookback).std()

   # df['volatility_norm'] = (df['volatility'] - df['volatility'].min()) / (df['volatility'].max() - df['volatility'].min())



# Fill NaN values (for the initial period where there's no rolling window data)
    df['volatility'].fillna(method='backfill', inplace=True)

    #print(volatility)
    
    # Merge daily volatility back into the original dataframe
    #df = df.merge(daily_volatility.rename('daily_volatility'), left_index=True, right_index=True, how='left')
    
    # Reset index to restore original structure
    df.reset_index(inplace=True)
    
    # Add necessary columns
    df['Datetime'] = pd.to_datetime(df['Date'])
    df['unix'] = df['Datetime'].astype('int64') // 10**9
    df['upper_barrier'] = df['Close'] * (1 + 3 * df['volatility'])
    df['lower_barrier'] = df['Close'] * (1 - 3 * df['volatility'])
    
    # Calculate lookback in seconds
    lookback_seconds = (lookback*10) * 3600
    df['endbarrier_unix'] = df['unix'] + (1.5*lookback_seconds)

    # Copy df to price_df and keep only 'unix' and 'Close'
    price_df = df[['unix', 'Close']].copy()

    # Convert price_df to a numpy array
    price_df_values = price_df.values

    def find_prices_in_range(start_unix, end_unix):
        mask = (price_df_values[:, 0] >= start_unix) & (price_df_values[:, 0] <= end_unix)
      #  mask = (price_df_values[:, 0])
        return price_df_values[mask, :]

    # Apply the function to each row
    df['prices_in_range'] = df.parallel_apply(lambda row: find_prices_in_range(row['unix'], row['endbarrier_unix']), axis=1)

    # Function to check if prices hit barriers and which hits first
    def price_barrier_check(row):
        lower_barrier = row['lower_barrier']
        upper_barrier = row['upper_barrier']
        prices_and_times = row['prices_in_range']  # This now includes both UNIX times and prices

        if prices_and_times.size == 0:
            return np.nan, np.nan, np.nan

        # Extract UNIX times and prices separately
        prices_unix_times = prices_and_times[:, 0]  # First column: UNIX times
        prices = prices_and_times[:, 1]  # Second column: prices

        # Find the hits on the barriers
        upper_hits = np.where(prices > upper_barrier)[0]
        lower_hits = np.where(prices < lower_barrier)[0]
        
        if upper_hits.size == 0 and lower_hits.size == 0:
            print('prices end:', prices[-1])

            end_pips = prices[-1] * 10000
            pct = row['pips'] / end_pips
            return np.nan, prices[-1], pct, np.nan

        if upper_hits.size > 0 and (lower_hits.size == 0 or prices_unix_times[upper_hits[0]] < prices_unix_times[lower_hits[0]]):
            # First upper hit happens before the first lower hit
            end_pips = prices[-1] * 10000
            pct = row['pips'] / end_pips
            return 1, prices[upper_hits[0]], pct, prices_unix_times[upper_hits[0]]

        if lower_hits.size > 0 and (upper_hits.size == 0 or prices_unix_times[lower_hits[0]] < prices_unix_times[upper_hits[0]]):
            # First lower hit happens before the first upper hit
            end_pips = prices[lower_hits[0]] * 10000
            pct = row['pips'] / end_pips


            return 0, prices[lower_hits[0]], pct, prices_unix_times[lower_hits[0]]


    '''
    # Function to find prices between unix and endbarrier_unix
    def find_prices_in_range(start_unix, end_unix):
        mask = (price_df_values[:, 0] >= start_unix) & (price_df_values[:, 0] <= end_unix)
      #  mask = (price_df_values[:, 0])
        return price_df_values[mask, 1]

    # Apply the function to each row
    df['prices_in_range'] = df.parallel_apply(lambda row: find_prices_in_range(row['unix'], row['endbarrier_unix']), axis=1)

    # Function to check if prices hit barriers and which hits first
    def price_barrier_check(row):
        lower_barrier = row['lower_barrier']
        upper_barrier = row['upper_barrier']
        prices = row['prices_in_range']
        pip_close = row['pips']
        
        # Use numpy to find the first occurrence of crossing the barriers
        if prices.size == 0:
            return np.nan, np.nan, np.nan
        
        upper_hits = np.where(prices > upper_barrier)[0]
        lower_hits = np.where(prices < lower_barrier)[0]
        
        if upper_hits.size == 0 and lower_hits.size == 0:
            print('prices end:', prices[-1])

            end_pips = prices[-1] *10000
            pct = pip_close / end_pips

            return np.nan, prices[-1], pct
        
        if upper_hits.size > 0 and (lower_hits.size == 0 or price_df_values[upper_hits[0], 0] < price_df_values[lower_hits[0], 0]):
            #pct = close / prices[upper_hits[0]]
            end_pips = prices[-1] *10000
            pct = pip_close / end_pips
            
            
            return 1, prices[upper_hits[0]], pct
        elif lower_hits.size > 0 and (upper_hits.size == 0 or price_df_values[lower_hits[0], 0] < price_df_values[upper_hits[0], 0]):
            #pct = close / prices[lower_hits[0]]
            end_pips = prices[lower_hits[0]] *10000
            pct = pip_close / end_pips

            return 0, prices[lower_hits[0]], pct

    # Apply the price_barrier_check function to each row
    '''
    df[['label', 'touch_price','pct', 'touch_time_unix']] = df.parallel_apply(price_barrier_check, axis=1, result_type='expand')
    
    end_time = time.time()
    runtime = end_time - start_time
    print(f'Function runtime: {runtime}')
    df1 = df[4000:5000]
    df1.to_csv('barriers_results.csv')
    
    return df


'''
def calculate_barriers_R(df, lookback):
    start_time = time.time()
    #new_bar_df['pips'] = new_bar_df['Close'] *10000
    
    # Ensure 'Date' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Set 'Date' as index for resampling
    df.set_index('Date', inplace=True)
    
    df['pct_change'] = df['Close'].pct_change()

# Calculate rolling standard deviation (volatility)
    df['volatility'] = df['pct_change'].rolling(window=lookback).std()

# Fill NaN values (for the initial period where there's no rolling window data)
    df['volatility'].fillna(method='backfill', inplace=True)

    #print(volatility)
    
    # Merge daily volatility back into the original dataframe
    #df = df.merge(daily_volatility.rename('daily_volatility'), left_index=True, right_index=True, how='left')
    
    # Reset index to restore original structure
    df.reset_index(inplace=True)
    
    # Add necessary columns
    df['Datetime'] = pd.to_datetime(df['Date'])
    df['unix'] = df['Datetime'].astype('int64') // 10**9
    df['upper_barrier'] = df['Close'] * (1 + 2 * df['volatility'])
    df['lower_barrier'] = df['Close'] * (1 - 2 * df['volatility'])
    
    # Calculate lookback in seconds
    lookback_seconds = (lookback*10) * 3600
    df['endbarrier_unix'] = df['unix'] + lookback_seconds

    # Copy df to price_df and keep only 'unix' and 'Close'
    price_df = df[['unix', 'Close']].copy()

    # Convert price_df to a numpy array
    price_df_values = price_df.values

    # Function to find prices between unix and endbarrier_unix
    def find_prices_in_range(start_unix, end_unix):
        mask = (price_df_values[:, 0] >= start_unix) & (price_df_values[:, 0] <= end_unix)
      #  mask = (price_df_values[:, 0])
        return price_df_values[mask, 1]

    # Apply the function to each row
    df['prices_in_range'] = df.parallel_apply(lambda row: find_prices_in_range(row['unix'], row['endbarrier_unix']), axis=1)

    # Function to check if prices hit barriers and which hits first
    def price_barrier_check(row):
        lower_barrier = row['lower_barrier']
        upper_barrier = row['upper_barrier']
        prices = row['prices_in_range']
        pip_close = row['pips']
        
        # Use numpy to find the first occurrence of crossing the barriers
        if prices.size == 0:
            return np.nan, np.nan, np.nan
        
        upper_hits = np.where(prices > upper_barrier)[0]
        lower_hits = np.where(prices < lower_barrier)[0]
        
        if upper_hits.size == 0 and lower_hits.size == 0:
            print('prices end:', prices[-1])

            end_pips = prices[-1] *10000
            pct = pip_close / end_pips

            return np.nan, prices[-1], pct
        
        if upper_hits.size > 0 and (upper_hits.size > lower_hits.size):
            #pct = close / prices[upper_hits[0]]
            end_pips = prices[-1] *10000
            pct = pip_close / end_pips
            
            
            return 1, prices[upper_hits[0]], pct
        elif lower_hits.size > 0 and (lower_hits.size > upper_hits.size):
            #pct = close / prices[lower_hits[0]]
            end_pips = prices[lower_hits[0]] *10000
            pct = pip_close / end_pips

            return 0, prices[lower_hits[0]], pct

    # Apply the price_barrier_check function to each row
    df[['label', 'touch_price','pct']] = df.parallel_apply(price_barrier_check, axis=1, result_type='expand')
    
    end_time = time.time()
    runtime = end_time - start_time
    print(f'Function runtime: {runtime}')
    df1 = df[4000:5000]
    df1.to_csv('barriers_results.csv')
    
    return df

'''
