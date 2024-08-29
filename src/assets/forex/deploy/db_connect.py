import os
import json
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine
import datetime

def insert_data(name):
    # Load JSON data
    with open(os.path.join('prediction', 'prediction_df.json'), 'r') as f:
        data = json.load(f)

    # Connect to MySQL
    try:
        with mysql.connector.connect(user='doadmin', password='AVNS_oW0kYA-LJsBz5pksVi4',
                                     host='tq-training-data-do-user-13042543-0.c.db.ondigitalocean.com',
                                     database='defaultdb', port=25060) as cnx:
            with cnx.cursor() as cursor:
                # Insert data into EURUSD_Live
                for record in data:
                    # Convert timestamp to MySQL date format
                    date_value = datetime.utcfromtimestamp(record['Date'] / 1000).strftime('%Y-%m-%d')
                    datehold_value = datetime.utcfromtimestamp(record['Datehold'] / 1000).strftime('%Y-%m-%d')
                    
                    query = """INSERT INTO EURUSD_Live (
                                    Date, Close, Volume, Close_AUDUSD, pips, `change`, pct_change, Datehold, day_of_week, 
                                    Middle_Band, Upper_Band, Lower_Band, Log_Returns, MACD, Signal_Line_MACD, RSI, 
                                    `K_percent`, `D_percent`, daily_return, direction, volume_direction, OBV, tenkan_sen, 
                                    kijun_sen, senkou_span_a, senkou_span_b, chikou_span, volatility, upper_barrier, 
                                    lower_barrier, prediction_proba_dwn, prediction_proba_up) 
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                                       %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                               ON DUPLICATE KEY UPDATE 
                               Close = VALUES(Close), 
                               Volume = VALUES(Volume), 
                               Close_AUDUSD = VALUES(Close_AUDUSD), 
                               pips = VALUES(pips), 
                               `change` = VALUES(`change`), 
                               pct_change = VALUES(pct_change), 
                               Datehold = VALUES(Datehold),
                               day_of_week = VALUES(day_of_week),
                               Middle_Band = VALUES(Middle_Band),
                               Upper_Band = VALUES(Upper_Band),
                               Lower_Band = VALUES(Lower_Band),
                               Log_Returns = VALUES(Log_Returns),
                               MACD = VALUES(MACD),
                               Signal_Line_MACD = VALUES(Signal_Line_MACD),
                               RSI = VALUES(RSI),
                               `K_percent` = VALUES(`K_percent`),
                               `D_percent` = VALUES(`D_percent`),
                               daily_return = VALUES(daily_return),
                               direction = VALUES(direction),
                               volume_direction = VALUES(volume_direction),
                               OBV = VALUES(OBV),
                               tenkan_sen = VALUES(tenkan_sen), 
                               kijun_sen = VALUES(kijun_sen),
                               senkou_span_a = VALUES(senkou_span_a), 
                               senkou_span_b = VALUES(senkou_span_b),
                               chikou_span = VALUES(chikou_span),
                               volatility = VALUES(volatility),
                               upper_barrier = VALUES(upper_barrier),
                               lower_barrier = VALUES(lower_barrier),
                               prediction_proba_dwn = VALUES(prediction_proba_dwn),
                               prediction_proba_up = VALUES(prediction_proba_up)
                               """
                    
                    # Ensure all fields are correctly passed
                    cursor.execute(query, (
                        date_value,  # Converted from timestamp to date string
                        record['Close'], 
                        record['Volume'], 
                        record['Close_AUDUSD'], 
                        record['pips'], 
                        record['change'], 
                        record['pct_change'], 
                        datehold_value,  # Converted from timestamp to date string
                        record['day_of_week'],
                        record['Middle_Band'],
                        record['Upper_Band'],
                        record['Lower_Band'],
                        record['Log_Returns'],
                        record['MACD'],
                        record['Signal_Line_MACD'],
                        record['RSI'],
                        record['K_percent'],
                        record['D_percent'],
                        record['daily_return'],
                        record['direction'],
                        record['volume_direction'],
                        record['OBV'],
                        record['tenkan_sen'],
                        record['kijun_sen'],
                        record['senkou_span_a'],
                        record['senkou_span_b'],
                        record['chikou_span'],
                        record['volatility'],
                        record['upper_barrier'],
                        record['lower_barrier'],
                        record['prediction_proba_dwn'],
                        record['prediction_proba_up']
                    ))

                cnx.commit()
    except mysql.connector.Error as err:
        print(f"Something went wrong: {err}")

# Example usage



def data_pull():
    '''
    cnx = mysql.connector.connect(user='doadmin', password='AVNS_oW0kYA-LJsBz5pksVi4',
                              host='tq-training-data-do-user-13042543-0.c.db.ondigitalocean.com',
                              database='defaultdb', port=25060)

        # Load the data into pandas DataFrames
    eurusd = pd.read_sql('SELECT * FROM EURUSD', cnx)
    usdchf = pd.read_sql('SELECT * FROM USDCHF', cnx)
    '''
    engine = create_engine('mysql+mysqlconnector://doadmin:AVNS_oW0kYA-LJsBz5pksVi4@tq-training-data-do-user-13042543-0.c.db.ondigitalocean.com:25060/defaultdb')

    #####
    eurusd = pd.read_sql('SELECT Date, Close as Close_EURUSD,Volume as Volume_EURUSD FROM EURUSD', engine)
    usdchf = pd.read_sql('SELECT Date, Close as Close_USDCHF FROM USDCHF', engine)
    merged = pd.merge(eurusd, usdchf, on='Date', how='left')
    merged['Close_USDCHF'] = merged['Close_USDCHF'].fillna(method='ffill')
    ########
    #eurusd = pd.read_sql('SELECT Date, Close as Close_EURUSD FROM EURUSD', engine)
    usdcad = pd.read_sql('SELECT Date, Close as Close_USDCAD FROM USDCAD', engine)
    merged = pd.merge(merged, usdcad, on='Date', how='left')
    merged['Close_USDCAD'] = merged['Close_USDCAD'].fillna(method='ffill')
    #######
    usdhkd = pd.read_sql('SELECT Date, Close as Close_USDHKD FROM USDHKD', engine)
    merged = pd.merge(merged, usdhkd, on='Date', how='left')
    merged['Close_USDHKD'] = merged['Close_USDHKD'].fillna(method='ffill')
    #######
    audusd = pd.read_sql('SELECT Date, Close as Close_AUDUSD FROM AUDUSD', engine)
    merged = pd.merge(merged, audusd, on='Date', how='left')
    merged['Close_AUDUSD'] = merged['Close_AUDUSD'].fillna(method='ffill')
    #######
    usdjpy = pd.read_sql('SELECT Date, Close as Close_USDJPY FROM USDJPY', engine)
    merged = pd.merge(merged, usdjpy, on='Date', how='left')
    merged['Close_USDJPY'] = merged['Close_USDJPY'].fillna(method='ffill')
    ########
    durable = pd.read_sql('SELECT date as Date, value as durableGoods FROM durableGoods', engine)
    durable['Date'] = pd.to_datetime(durable['Date'])
    merged = pd.merge_asof(merged, durable, on='Date', direction='nearest')
    merged['durableGoods'] = merged['durableGoods'].fillna(method='ffill')
    ########

    
    merged.to_csv('merged.csv')
    
    return merged  

def add_merged():
    merged = pd.read_csv('merged.csv')
    merged['Date'] = pd.to_datetime(merged['Date'])  # Convert 'Date' to datetime

    indicator_list = ['15Yr_Fixed','30Yr_Fixed','CPI','GDP','Production_Total_Index','Yields_COD','consumerSentiment','federalFunds','inflation','inflationRate','initialClaims','nominalPotentialGDP','rates_CreditCards','realGDP','realGDPPerCapita', 'retailMoneyFunds', 'retailSales']

    engine = create_engine('mysql+mysqlconnector://doadmin:AVNS_oW0kYA-LJsBz5pksVi4@tq-training-data-do-user-13042543-0.c.db.ondigitalocean.com:25060/defaultdb')
     
    for i in indicator_list:
        data = pd.read_sql(f'SELECT date as Date, value as {i} FROM {i}', engine)
        data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime
        merged = pd.merge_asof(merged, data, on='Date', direction='nearest')
        merged[i] = merged[i].fillna(method='ffill')

    merged.to_csv('merged.csv')

    return merged

    return merged
    




if __name__ == "__main__":

    #print(data_pull())

    # Example usage
    insert_data('EURUSD_Live')

    
    #print(add_merged())
    
    '''
    path = 'updated_data/econ'
    dir_list = os.listdir(path)

    for s in dir_list:
        s = s.split(".")
        insert_data(s[0])
    '''
