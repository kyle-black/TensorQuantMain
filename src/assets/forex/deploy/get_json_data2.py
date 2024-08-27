import requests
import json
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value
import os
import pandas as pd
import json
#import mysql
execution_counter = Value('i', 0)




#url ="https://financialmodelingprep.com/api/v3/historical-chart/1min/ALIUSD?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey=3e17d2b777a13feee4c1243985cdc7c4"
#resp = requests.get(url)
#print(resp.json())

def get_json_from_url(symbol):
    
        start_date = datetime.strptime('2024-01-01', '%Y-%m-%d')
        end_date = start_date + timedelta(days=3)
        final_date = datetime.strptime('2024-08-26', '%Y-%m-%d')
        
        #global execution_counter

        execution_counter = 0

        data_list =[]
        while start_date <= final_date:
            execution_counter += 1

            if execution_counter >= 299:
                time.sleep(65)
                execution_counter.value = 0
                 
            print( f'retreving {symbol} start:{start_date} end:{end_date}')

            #start_date = ('2014-01-01', '%Y-%m-%d')
           # start_date =start_date.strftime('%Y-%m-%d')
            #url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{symbol}?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey=3e17d2b777a13feee4c1243985cdc7c4"
            
            url =f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{symbol}?from={start_date}&to={end_date}&apikey=3e17d2b777a13feee4c1243985cdc7c4"
            response = requests.get(url)

            execution_counter += 1
            print(response.json())            
        
            #if response.content:
            data = response.json()

            start_date += timedelta(days=3)
            end_date += timedelta(days=3)
           # data_list.append(data)
            for i in data:
                
                with open(f'updated_data/{symbol}.json', 'a') as f:
                    json.dump(i, f)
                   # f.write('\n')
           # print(data)    
            #return data

'''
def data_pull():
    cnx = mysql.connector.connect(user='doadmin', password='AVNS_oW0kYA-LJsBz5pksVi4',
                              host='tq-training-data-do-user-13042543-0.c.db.ondigitalocean.com',
                              database='defaultdb', port=25060)

        # Load the data into pandas DataFrames
    eurusd = pd.read_sql('SELECT * FROM EURUSD', cnx)
    usdchf = pd.read_sql('SELECT * FROM USDCHF', cnx)

    # Merge the two DataFrames on the 'date' column
    merged = pd.merge(eurusd, usdchf, on='date', how='left')

    # Fill in missing values in the 'value' column of the USDCHF data
    # with the previous non-null value in the same column
    merged['Close_usdchf'] = merged['close'].fillna(method='ffill')

    return merged     

if __name__ == "__main__":

    get_json_from_url('EURUSD')
   #get_json_from_url(symbol_list=['EURUSD'])#'GBPUSD','USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD'])

    #commod = ['PLUSD','GCUSD','SIUSD','NGUSD', 'CLUSD','HGUSD','PAUSD','ALIUSD']
    #symbol_list = ['ALIUSD']#,'USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD']
    #symbol_list = ['']#,'USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD']
   # data_dict = {}
    #for symbol in commod:

        
     #   print(f'adding {symbol}')
     #   data = get_json_from_url(symbol)

       # data_dict[symbol] = data

        # Save the dictionary to a JSON file
      #  with open(f'{symbol}.json', 'w') as f:
       #     json.dump(data, f)
    #df = pd.DataFrame(data_dict)

    #df.to_csv('commod.csv')
'''

    