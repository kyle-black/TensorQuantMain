import requests
import json
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value
import os
import pandas as pd
import json
execution_counter = Value('i', 0)




#url ="https://financialmodelingprep.com/api/v3/historical-chart/1min/ALIUSD?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey=3e17d2b777a13feee4c1243985cdc7c4"
#resp = requests.get(url)
#print(resp.json())

def get_json_from_url(symbol):
    
        
        url = f'https://financialmodelingprep.com/api/v4/economic?name={symbol}&apikey=3e17d2b777a13feee4c1243985cdc7c4'
        response = requests.get(url)

       
        print(response.json())            
 
        data = response.json()

        
        
            
        with open(f'updated_data/econ/{symbol}.json', 'a') as f:
            json.dump(data, f)
                #f.write('\n')
         

if __name__ == "__main__":
   
    econ=['GDP', 'realGDP', 'nominalPotentialGDP', 'realGDPPerCapita', 'federalFunds', 'CPI', 'inflationRate', 'inflation', 'retailSales', 'consumerSentiment', 'durableGoods', 'unemploymentRate', 'totalNonfarmPayroll', 'initialClaims', 'industrialProductionTotalIndex', 'newPrivatelyOwnedHousingUnitsStartedTotalUnits', 'totalVehicleSales', 'retailMoneyFunds', 'smoothedUSRecessionProbabilities', '3MonthOr90DayRatesAndYieldsCertificatesOfDeposit', 'commercialBankInterestRateOnCreditCardPlansAllAccounts', '30YearFixedRateMortgageAverage', '15YearFixedRateMortgageAverage']


    
    
    for symbol in econ:

        
        print(f'adding {symbol}')
        data = get_json_from_url(symbol)