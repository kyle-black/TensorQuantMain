import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay, DateOffset, WeekOfMonth, LastWeekOfMonth
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import Day, BMonthEnd
import time
from pandarallel import pandarallel
pandarallel.initialize()



'''
def new_apply_triple_barrier(df, pt_sl, endbar, asset):
    """
    Apply the triple barrier method to label events.

    Parameters:
    df: DataFrame with price data.
    pt_sl: List of multipliers for profit taking and stop-loss.
    num_days_active: Number of days the barrier should be kept active.

    Returns:
    DataFrame with events labeled.
    """
    
    if asset is not None:
        close ='Close'
     #   high = 'High'
     #   low = 'Low'
    else: 
        close='Close' 
      #  high='High'
      #  low ='Low' 
    



    df = df.loc[~df.Date.duplicated(keep='first')]
   # df.Date = df.Date.astype('int')

    #print(df.Date)
    #df.set_index('Date', inplace=True)

    df.index = pd.to_datetime(df['Date'])
    #df.index = pd.to_datetime(df.Date, unit='s')
    

    print('dataframeindex:', df.index)
   # endbar = endbar + 50
    # Compute rolling daily volatility
    #rolling_window = 72 # Example window size, you can adjust this
    daily_volatility = df[close].pct_change().std()

   # print('daily volit:', daily_volatility)

    barriers = pd.DataFrame(index=df.index)
    bar_count = 0
    for timestamp, data in df.iterrows():
        bar_count += 1
        price = data[close]
        volatility = daily_volatility

        upper_barrier = price * (1 + pt_sl[0] * (2*volatility))
        lower_barrier = price * (1 - pt_sl[1] * (2*volatility))

        barriers.at[timestamp, 'upper_barrier'] = upper_barrier
        barriers.at[timestamp, 'lower_barrier'] = lower_barrier

        t1_date = timestamp + pd.Timedelta(days=30)
        t1_date = min(t1_date, df.index[-1])
        
        barriers.at[timestamp, 't1'] = t1_date

        df_temp = df.loc[timestamp:].iloc[1:]


        try:
           # end_bar = df.iloc[(bar_count + endbar-1)].name

           # print('endbar!!!!:',end_bar)
           # print('endbartype:',type(end_bar))

            touch_upper = df_temp[df_temp[close] >= upper_barrier].index.min()
            touch_lower = df_temp[df_temp[close] <= lower_barrier].index.min()

            barriers.at[timestamp, 'touch_upper'] = touch_upper
            barriers.at[timestamp, 'touch_lower'] = touch_lower
            
          #  df['end_bar'] =end_bar
        ####### New LOGIC for barrier touching
            
            if (touch_upper < touch_lower) and (touch_upper < t1_date):
                barriers.at[timestamp, 'label'] = 1

            elif (touch_lower < touch_upper) and (touch_lower < t1_date):
                barriers.at[timestamp, 'label'] = -1
            
            else:
                barriers.at[timestamp, 'label'] = 0
        except:
            touch_upper = df_temp[df_temp[close] >= upper_barrier].index.min()
            touch_lower = df_temp[df_temp[close] <= lower_barrier].index.min()

            barriers.at[timestamp, 'touch_upper'] = touch_upper
            barriers.at[timestamp, 'touch_lower'] = touch_lower

            barriers.at[timestamp,'label'] =  np.nan
         
    ###############################################
    ####### New LOGIC for barrier touching
        
        if (touch_upper < touch_lower):
            barriers.at[timestamp, 'label'] = 1

        elif (touch_lower < touch_upper):
            barriers.at[timestamp, 'label'] = -1
        
 #       else:
  #          barriers.at[timestamp, 'label'] = 0
        
    ###############################################

    df_merged = df.join(barriers, how='left')
    df_merged.to_csv('sanity_check_72.csv')
    return df_merged


def calculate_barriers(row, df, pt_sl, time_, close='Close'):
    price = row[close]
    daily_volatility = df[close].pct_change().std()
    volatility = daily_volatility

    upper_barrier = price * (1 + pt_sl[0] * (2*volatility))
    lower_barrier = price * (1 - pt_sl[1] * (2*volatility))

    t1_date = row.name + pd.Timedelta(hours=time_)
    t1_date = min(t1_date, df.index[-1])

    df_temp = df.loc[row.name:].iloc[1:]

    touch_upper = df_temp[df_temp[close] >= upper_barrier].index.min()
    touch_lower = df_temp[df_temp[close] <= lower_barrier].index.min()

    if (touch_upper < touch_lower) and (touch_upper < t1_date):
        label = 1
    elif (touch_lower < touch_upper) and (touch_lower < t1_date):
        label = -1
    else:
        label = 0

    return pd.Series([upper_barrier, lower_barrier, t1_date, touch_upper, touch_lower, label], index=['upper_barrier', 'lower_barrier', 't1', 'touch_upper', 'touch_lower', 'label'])

df[['upper_barrier', 'lower_barrier', 't1', 'touch_upper', 'touch_lower', 'label']] = df.apply(calculate_barriers, axis=1)
'''

'''

def apply_triple_barrier_P(df, pt_sl, num_days_active):
    """
    Apply the triple barrier method to label events.

    Parameters:
    df: DataFrame with price data.
    pt_sl: List of multipliers for profit taking and stop-loss.
    num_days_active: Number of days the barrier should be kept active.

    Returns:
    DataFrame with events labeled.
    """
    df.index = pd.to_datetime(df.Date)
    
    # Compute rolling daily volatility
    rolling_window = 48  # Example window size, you can adjust this
    daily_volatility = df['Close'].pct_change().rolling(window=rolling_window).std()

    barriers = pd.DataFrame(index=df.index)

    for timestamp, data in df.iterrows():
        price = data['Close']
        volatility = daily_volatility.loc[timestamp]

        upper_barrier = price * (1 + pt_sl[0] * (1*volatility))
        lower_barrier = price * (1 - pt_sl[1] * (1*volatility))

        barriers.at[timestamp, 'upper_barrier'] = upper_barrier
        barriers.at[timestamp, 'lower_barrier'] = lower_barrier

        t1_date = timestamp + pd.Timedelta(hours=num_days_active)
        t1_date = min(t1_date, df.index[-1])
        
        barriers.at[timestamp, 't1'] = t1_date

        df_temp = df.loc[timestamp:].iloc[1:]

        touch_upper = df_temp[df_temp['High'] >= upper_barrier].index.min()
        touch_lower = df_temp[df_temp['Low'] <= lower_barrier].index.min()

        barriers.at[timestamp, 'touch_upper'] = touch_upper
        barriers.at[timestamp, 'touch_lower'] = touch_lower

        if touch_upper and touch_upper < t1_date:
            barriers.at[timestamp, 'label'] = 1
        elif touch_lower and touch_lower < t1_date:
            barriers.at[timestamp, 'label'] = -1
        else:
            barriers.at[timestamp, 'label'] = 0 

    df_merged = df.join(barriers, how='left')
    df_merged.to_csv('sanity_check.csv')
    return df_merged
'''

'''
def calculate_barriers_R(df, lookback):
    volatility = df['Close'].pct_change().rolling(window=1000).std()
    #volatility = df['Close'].pct_change().std()
    date_index = df.index
    
    df['Datetime'] = pd.to_datetime(df['Date'])
    df['unix'] = df['Datetime'].astype('int64') // 10**9
    df['upper_barrier'] = df['Close'] * (1 +1 * (1*volatility))
    df['lower_barrier'] = df['Close'] * (1 -1 * (1*volatility))

    lookback_hours = pd.Timedelta(hours=lookback)
    lookback_unix = lookback_hours / pd.Timedelta('1s')

    df['endbarrier_unix'] = df['unix'] + lookback_unix
    df = df[['unix','Close','endbarrier_unix','upper_barrier','lower_barrier']]
    arr = df.to_numpy()

    next_arr = np.roll(arr, -1, axis=0)

    print('next arr:', next_arr)

    # Check if the Close value of the next row is greater than the upper barrier or less than the lower barrier
    upper_touches = (next_arr[:-1, 1] > arr[:-1, 3]) & (next_arr[:-1, 0] <= arr[:-1, 2])
    lower_touches = (next_arr[:-1, 1] < arr[:-1, 4]) & (next_arr[:-1, 0] <= arr[:-1, 2])
    #lower_touches = (next_arr[:-1, 1] < arr[:-1, 4]) & (next_arr[:-1, 0] <= arr[:-1, 2])
 
    

    # Create a new array that contains 1 where the upper barrier is touched first, -1 where the lower barrier is touched first, and 0 where the end barrier is reached before either the upper or lower barrier is touched
    labels = np.where(upper_touches, 1, np.where(lower_touches, -1, 0))

    # Append a default label to the end of the labels array
    labels = np.append(labels, 0)

    # Add the labels array as a new column to the original array
    arr = np.column_stack((arr, labels))

    # Add the Close price of the upper barrier touch, lower barrier touch, or end barrier close
    touch_price = np.where(upper_touches, next_arr[:-1, 1], np.where(lower_touches, next_arr[:-1, 1], arr[:-1, 1]))
    touch_price = np.append(touch_price, arr[-1, 1])
    arr = np.column_stack((arr, touch_price))

    df = pd.DataFrame(arr, columns=['unix','Close','endbarrier_unix','upper_barrier','lower_barrier', 'label', 'touch_price'])
    df.drop('Close', axis =1,inplace=True)
    df.index = date_index

    print('barrier df:', df.columns)

    return df
'''
'''
def calculate_barriers_R(df, lookback):
    # Calculate volatility
    df['pct_change'] = df['Close'].pct_change()
    volatility = df['pct_change'].rolling(window=1000).std()
    
    # Add necessary columns
    df['Datetime'] = pd.to_datetime(df['Date'])
    df['unix'] = df['Datetime'].astype('int64') // 10**9
    df['upper_barrier'] = df['Close'] * (1 + volatility)
    df['lower_barrier'] = df['Close'] * (1 - volatility)
    
    # Calculate lookback in seconds
    lookback_seconds = lookback * 3600
    df['endbarrier_unix'] = df['unix'] + lookback_seconds

    # Create a new DataFrame for future values lookup
    future_df = df[['unix', 'Close']].copy()
    future_df.columns = ['future_unix', 'future_close']
    
    # Merge to get the closest future_close before endbarrier_unix
    df = pd.merge_asof(df, future_df, left_on='endbarrier_unix', right_on='future_unix', direction='forward')

    # Initialize columns for results
    df['label'] = 0
    df['touch_price'] = df['future_close']

    # Vectorized touch condition checks
    mask_upper = df['future_close'] > df['upper_barrier']
    mask_lower = df['future_close'] < df['lower_barrier']

    # Determine labels and touch prices
    df.loc[mask_upper, 'label'] = 1
    df.loc[mask_lower, 'label'] = -1
    df.loc[mask_upper, 'touch_price'] = df['future_close'][mask_upper]
    df.loc[mask_lower, 'touch_price'] = df['future_close'][mask_lower]
   # print(df['Close','future_close','upper_barrier','lower_barrier'])
    # Drop unnecessary columns
    df.drop([  'future_unix', 'future_close'], axis=1, inplace=True)
    
    df = df.dropna()
    #df.to_csv('barrier_check.csv')
    df.to_parquet('barrier_check.parquet')

    df.to_csv('barriercheck.csv')

    return df
'''

'''

def calculate_barriers_R(df, lookback):
    # Calculate volatility
    df['pct_change'] = df['Close'].pct_change()
    volatility = df['pct_change'].rolling(window=1000).std()
    
    # Add necessary columns
    df['Datetime'] = pd.to_datetime(df['Date'])
    df['unix'] = df['Datetime'].astype('int64') // 10**9
    df['upper_barrier'] = df['Close'] * (1 + volatility)
    df['lower_barrier'] = df['Close'] * (1 - volatility)
    
    # Calculate lookback in seconds
    lookback_seconds = lookback * 3600
    df['endbarrier_unix'] = df['unix'] + lookback_seconds

    # Calculate lookback periods
    lookback_periods = int(lookback_seconds / (df['unix'].iloc[1] - df['unix'].iloc[0]))

    # Create a rolling window function to check barriers
    def check_label(x):
        start = df.index.get_loc(x.index[0])
        upper_barrier = df['upper_barrier'].iloc[start]
        lower_barrier = df['lower_barrier'].iloc[start]
        close_prices = x.values

        if (close_prices > upper_barrier).any():
            return 1
        elif (close_prices < lower_barrier).any():
            return -1
        else:
            return 0

    def check_touch_price(x):
        start = df.index.get_loc(x.index[0])
        upper_barrier = df['upper_barrier'].iloc[start]
        lower_barrier = df['lower_barrier'].iloc[start]
        close_prices = x.values

        if (close_prices > upper_barrier).any():
            return close_prices[close_prices > upper_barrier][0]
        elif (close_prices < lower_barrier).any():
            return close_prices[close_prices < lower_barrier][0]
        else:
            return np.nan

    # Apply the rolling window for label and touch price
    df['label'] = df['Close'].rolling(window=lookback_periods, min_periods=1).apply(check_label, raw=False)
    df['touch_price'] = df['Close'].rolling(window=lookback_periods, min_periods=1).apply(check_touch_price, raw=False)

    # Drop unnecessary columns and rows with NaN touch prices
   # df.drop(['pct_change'], axis=1, inplace=True)
    df.dropna(subset=['touch_price'], inplace=True)

    # Save to parquet and csv
    df.to_parquet('barrier_check.parquet')
    df.to_csv('barrier_check.csv')

    return df


def calculate_barriers_R(df, lookback):
    # Calculate volatility
    #df = df[:]
    df['pct_change'] = df['Close'].pct_change()
    volatility = df['pct_change'].rolling(window=lookback).std()
    
    # Add necessary columns
    df['Datetime'] = pd.to_datetime(df['Date'])
    df['unix'] = df['Datetime'].astype('int64') // 10**9
    df['upper_barrier'] = df['Close'] * (1 + 2 * volatility)
    df['lower_barrier'] = df['Close'] * (1 - 2 * volatility)
    
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
            return 1, np.nan
        
        upper_hits = np.where(prices > upper_barrier)[0]
        lower_hits = np.where(prices < lower_barrier)[0]
        
        if upper_hits.size == 0 and lower_hits.size == 0:
            return 1, np.nan
        
        if upper_hits.size > 0 and (lower_hits.size == 0 or upper_hits[0] < lower_hits[0]):
            return 2, prices[upper_hits[0]]
        elif lower_hits.size > 0 and (upper_hits.size == 0 or lower_hits[0] < upper_hits[0]):
            return 0, prices[lower_hits[0]]

    # Apply the price_barrier_check function to each row
    df[['label', 'touch_price']] = df.parallel_apply(price_barrier_check, axis=1, result_type='expand')
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"Non-numeric columns detected: {non_numeric_cols}")
        print("Please handle these columns before proceeding.")

    df.to_csv('updated_df3.csv')
        
    return df


def calculate_barriers_R(df, lookback):
    # Calculate volatility
    #df = df[:]
    df['pct_change'] = df['Close'].pct_change()
    volatility = df['pct_change'].rolling(window=lookback).std()
    
    # Add necessary columns
    df['Datetime'] = pd.to_datetime(df['Date'])
    df['unix'] = df['Datetime'].astype('int64') // 10**9
    df['upper_barrier'] = df['Close'] * (1 + 2 * volatility)
    df['lower_barrier'] = df['Close'] * (1 - 2 * volatility)
    
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
  #  df['prices_in_range'] = df.apply(lambda row: find_prices_in_range(row['unix'], row['endbarrier_unix']), axis=1)

    barrier_df = [[df.index,'unix','endbarrier_unix','upper_barrier', 'lower_barrier']]

    barrier_df_values = barrier_df.values


    arr =np.where(price_df_values)
    
       
    # Function to check if prices hit barriers and which hits first
    def price_barrier_check(row):
        lower_barrier = row['lower_barrier']
        upper_barrier = row['upper_barrier']
        prices = row['prices_in_range']
        
        # Use numpy to find the first occurrence of crossing the barriers
        if prices.size == 0:
            return 1, np.nan
        
        upper_hits = np.where(prices > upper_barrier)[0]
        lower_hits = np.where(prices < lower_barrier)[0]
        
        if upper_hits.size == 0 and lower_hits.size == 0:
            return 1, np.nan
        
        if upper_hits.size > 0 and (lower_hits.size == 0 or upper_hits[0] < lower_hits[0]):
            return 2, prices[upper_hits[0]]
        elif lower_hits.size > 0 and (upper_hits.size == 0 or lower_hits[0] < upper_hits[0]):
            return 0, prices[lower_hits[0]]

    # Apply the price_barrier_check function to each row
    df[['label', 'touch_price']] = df.apply(price_barrier_check, axis=1, result_type='expand')
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"Non-numeric columns detected: {non_numeric_cols}")
        print("Please handle these columns before proceeding.")

    df.to_csv('updated_df3.csv')
        
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
    
    # Calculate daily percentage change and daily volatility
    df['pct_change'] = df['Close'].pct_change()
    daily_volatility = df['pct_change'].resample('d').std().mean()

    print(daily_volatility)
    
    # Merge daily volatility back into the original dataframe
    #df = df.merge(daily_volatility.rename('daily_volatility'), left_index=True, right_index=True, how='left')
    
    # Reset index to restore original structure
    df.reset_index(inplace=True)
    
    # Add necessary columns
    df['Datetime'] = pd.to_datetime(df['Date'])
    df['unix'] = df['Datetime'].astype('int64') // 10**9
    df['upper_barrier'] = df['Close'] * (1 + 1 * daily_volatility)
    df['lower_barrier'] = df['Close'] * (1 - 1 * daily_volatility)
    
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
        pip_close = row['pips']
        
        # Use numpy to find the first occurrence of crossing the barriers
        if prices.size == 0:
            return 0, np.nan, np.nan
        
        upper_hits = np.where(prices > upper_barrier)[0]
        lower_hits = np.where(prices < lower_barrier)[0]
        
        if upper_hits.size == 0 and lower_hits.size == 0:
            print('prices end:', prices[-1])

            end_pips = prices[-1] *10000
            pct = pip_close / end_pips

            return 0, prices[-1], pct
        
        if upper_hits.size > 0 and (lower_hits.size == 0 or upper_hits[0] < lower_hits[0]):
            #pct = close / prices[upper_hits[0]]
            end_pips = prices[-1] *10000
            pct = pip_close / end_pips
            
            
            return 1, prices[upper_hits[0]], pct
        elif lower_hits.size > 0 and (upper_hits.size == 0 or lower_hits[0] < upper_hits[0]):
            #pct = close / prices[lower_hits[0]]
            end_pips = prices[lower_hits[0]] *10000
            pct = pip_close / end_pips

            return -1, prices[lower_hits[0]], pct

    # Apply the price_barrier_check function to each row
    df[['label', 'touch_price','pct']] = df.parallel_apply(price_barrier_check, axis=1, result_type='expand')
    
    end_time = time.time()
    runtime = end_time - start_time
    print(f'Function runtime: {runtime}')
    
    return df, daily_volatility

    

