import pandas as pd
import numpy as np
import time
import multiprocessing

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
    price_df = df[['unix', 'Close']].to_numpy()

    # Initialize arrays to store results
    labels = np.zeros(len(df), dtype=int)
    touch_prices = np.full(len(df), np.nan)

    for i in range(len(df)):
       # print('date:',df['Datetime'].iloc[i])
        start_unix = df['unix'].iloc[i]
        end_unix = df['endbarrier_unix'].iloc[i]
        upper_barrier = df['upper_barrier'].iloc[i]
        lower_barrier = df['lower_barrier'].iloc[i]

        # Find indices within the range
        mask = (price_df[:, 0] >= start_unix) & (price_df[:, 0] <= end_unix)
        prices_in_range = price_df[mask, 1]

        if prices_in_range.size == 0:
            continue

        # Check for barrier hits
        upper_hits = np.where(prices_in_range > upper_barrier)[0]
        lower_hits = np.where(prices_in_range < lower_barrier)[0]

        if upper_hits.size > 0 and (lower_hits.size == 0 or upper_hits[0] < lower_hits[0]):
            labels[i] = 1
            touch_prices[i] = prices_in_range[upper_hits[0]]
        elif lower_hits.size > 0 and (upper_hits.size == 0 or lower_hits[0] < upper_hits[0]):
            labels[i] = -1
            touch_prices[i] = prices_in_range[lower_hits[0]]

    df['label'] = labels
    df['touch_price'] = touch_prices

    df.to_csv('updated_df.csv')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(df)
    return df




if __name__ == '__main__':
    period = 100000
    # Example usage
    data = {
    'Date': pd.date_range(start='2022-01-01', periods=period, freq='h'),
    'Close': np.random.rand(period) * period
    }
    df = pd.DataFrame(data)
    lookback = 4  # 24 hours lookback period
    #result_df = calculate_barriers_R(df, lookback)
    #print(result_df)
    starttime = time.time()
    processes = []
    #for i in range(0,10):
    p = multiprocessing.Process(target=calculate_barriers_R, args=(df,lookback))
    processes.append(p)
    p.start()
        
    for process in processes:
        process.join()
        
    print('That took {} seconds'.format(time.time() - starttime))