import requests
import json
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value
import os
execution_counter = Value('i', 0)

def get_json_from_url(symbol):
    try:
        start_date = datetime.strptime('2010-01-01', '%Y-%m-%d')
        end_date = start_date + timedelta(days=3)
        final_date = datetime.strptime('2010-03-01', '%Y-%m-%d')
        
        global execution_counter

        while start_date <= final_date:
            url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{symbol}?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey=3e17d2b777a13feee4c1243985cdc7c4"
            response = requests.get(url)
            
            if response.content:
                data = response.json()
            else:
                print(f"No data returned for {symbol} from {start_date} to {end_date}")
                return
            
            print(f'retreving {symbol} start:{start_date} end:{end_date}')
            
            # Check if file exists, if not, create it with an empty list
            filename = f'updated_data/commod/{symbol}.json'
            if not os.path.exists(filename):
                with open(filename, 'w') as f:
                    json.dump([], f)

            # Read existing data from file
            with open(filename, 'r') as f:
                existing_data = json.load(f)

            # Append new data to existing data
            existing_data.extend(data)

            # Write updated data back to file
            with open(filename, 'w') as f:
                json.dump(existing_data, f)

            # Increment dates by 3 days
            start_date += timedelta(days=3)
            end_date += timedelta(days=3)

            # Increment execution counter
            with execution_counter.get_lock():
                execution_counter.value += 1
                print(execution_counter.value)

                # If execution counter hits 299, pause for 60 seconds and reset counter
                if execution_counter.value >= 299:
                    time.sleep(70)
                    execution_counter.value = 0
    except Exception as e:
        print(f"An error occurred in thread {symbol}: {e}")
def main(symbol_list):
    with ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(get_json_from_url, symbol_list)
if __name__ == "__main__":
   #get_json_from_url(symbol_list=['EURUSD'])#'GBPUSD','USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD'])

    commod = ['PLUSD','GCUSD','SIUSD','NGUSD', 'CLUSD','HGUSD','PAUSD','ALIUSD']
    #symbol_list = ['EURUSD','GBPUSD']#,'USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD']
    #symbol_list = ['NZDJPY']#,'USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD']
    main(commod)