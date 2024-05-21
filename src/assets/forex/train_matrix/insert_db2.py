import os
import json
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine

def insert_data(name):
    # Load JSON data
    with open(os.path.join('updated_data', 'econ', f'{name}.json'), 'r') as f:
        data = json.load(f)

    # Connect to MySQL
    try:
        with mysql.connector.connect(user='doadmin', password='AVNS_oW0kYA-LJsBz5pksVi4',
                                    host='tq-training-data-do-user-13042543-0.c.db.ondigitalocean.com',
                                    database='defaultdb', port=25060) as cnx:
            with cnx.cursor() as cursor:
                # Create table
                create_table_query = f"""CREATE TABLE IF NOT EXISTS {name} (
                                            date DATE PRIMARY KEY,
                                            value FLOAT
                                        )"""
                cursor.execute(create_table_query)

                # Insert data
                for record in data:
                    query = f"""INSERT INTO {name} (date, value) VALUES (%s, %s) 
                                ON DUPLICATE KEY UPDATE value = VALUES(value)"""
                    cursor.execute(query, (record['date'], record['value']))

                cnx.commit()
    except mysql.connector.Error as err:
        print(f"Something went wrong: {err}")


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
    merged['Close_USDPY'] = merged['Close_USDJPY'].fillna(method='ffill')
    ########


    merged.to_csv('merged.csv')
    
    return merged     



if __name__ == "__main__":

    print(data_pull())
    '''
    path = 'updated_data/econ'
    dir_list = os.listdir(path)

    for s in dir_list:
        s = s.split(".")
        insert_data(s[0])
    '''
