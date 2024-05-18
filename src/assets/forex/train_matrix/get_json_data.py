import requests
import json
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value
execution_counter = Value('i', 0)

def get_json_from_url(symbol):
    start_date = datetime.strptime('2018-07-31', '%Y-%m-%d')
    
    end_date = start_date + timedelta(days=3)
    final_date = datetime.strptime('2024-01-03', '%Y-%m-%d')
    
    global execution_counter

    while start_date <= final_date:
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{symbol}?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey=3e17d2b777a13feee4c1243985cdc7c4"
        response = requests.get(url)
        data = response.json()
        print(f'retreving {symbol} start:{start_date} end:{end_date}')
        
        # Append data to JSON file
        with open(f'updated_data/train_data/redo/{symbol}.json', 'a') as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')

        # Increment dates by 10 days
        start_date += timedelta(days=3)
        end_date += timedelta(days=3)

        # Increment execution counter
        with execution_counter.get_lock():
            execution_counter.value += 1
            print(execution_counter.value)

            # If execution counter hits 240, pause for 60 seconds and reset counter
            if execution_counter.value >= 240:
                time.sleep(3)
                execution_counter.value = 0
def main(symbol_list):
    with ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(get_json_from_url, symbol_list)
if __name__ == "__main__":
   #get_json_from_url(symbol_list=['EURUSD'])#'GBPUSD','USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD'])


    #symbol_list = ['EURUSD','GBPUSD']#,'USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD']
    symbol_list = ['NZDJPY']#,'USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD']
    main(symbol_list)
