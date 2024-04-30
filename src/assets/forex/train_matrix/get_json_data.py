import requests
import json
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value


execution_counter = Value('i', 0)

def get_json_from_url(symbol):
    start_date = datetime.strptime('2015-01-01', '%Y-%m-%d')
    end_date = start_date + timedelta(days=10)
    final_date = datetime.strptime('2020-01-15', '%Y-%m-%d')

    global execution_counter

    #for symbol in symbol_list:
    while start_date <= final_date:
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{symbol}?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey=3e17d2b777a13feee4c1243985cdc7c4"
        response = requests.get(url)
        data = response.json()

        print(f'retreving {symbol} start:{start_date} end:{end_date}')

        # Save or append data to JSON file
        try:
            with open(f'updated_data/{symbol}_1.json', 'r') as f:
                if f.read().strip():  # Check if file is not empty
                    f.seek(0)  # Reset file pointer to beginning
                    existing_data = json.load(f)
                    combined_data = existing_data + data
                else:
                    combined_data = data
        except FileNotFoundError:
            combined_data = data

        with open(f'updated_data/{symbol}_1.json', 'w') as f:
            json.dump(combined_data, f)

        # Increment dates by 10 days
        start_date += timedelta(days=10)
        end_date += timedelta(days=10)

        # Increment execution counter
        # Increment execution counter
        with execution_counter.get_lock():
            execution_counter.value += 1

            # If execution counter hits 240, pause for 60 seconds and reset counter
            if execution_counter.value >= 240:
                time.sleep(10)
                execution_counter.value = 0



def main(symbol_list):
    with ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(get_json_from_url, symbol_list)



if __name__ == "__main__":
   #get_json_from_url(symbol_list=['EURUSD'])#'GBPUSD','USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD'])


    symbol_list = ['EURUSD','GBPUSD']#,'USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD']
    main(symbol_list)
  