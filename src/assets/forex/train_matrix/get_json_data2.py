import requests
import json
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value
import os
execution_counter = Value('i', 0)

def get_json_from_url(symbol):
    
        start_date = datetime.strptime('2010-01-01', '%Y-%m-%d')
        end_date = start_date + timedelta(days=3)
        final_date = datetime.strptime('2010-03-01', '%Y-%m-%d')
        
        global execution_counter

        while start_date <= final_date:
            url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{symbol}?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey=3e17d2b777a13feee4c1243985cdc7c4"
            response = requests.get(url)
            
            if response.content:
                data = response.json()

                print(data)
            
if __name__ == "__main__":
   #get_json_from_url(symbol_list=['EURUSD'])#'GBPUSD','USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD'])

    commod = ['PLUSD','GCUSD','SIUSD','NGUSD', 'CLUSD','HGUSD','PAUSD','ALIUSD']
    #symbol_list = ['EURUSD','GBPUSD']#,'USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD']
    #symbol_list = ['NZDJPY']#,'USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD']
    get_json_from_url(commod)