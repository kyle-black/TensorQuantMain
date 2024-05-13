from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float, DateTime, MetaData, VARCHAR
from sqlalchemy import BigInteger
import os
from sqlalchemy import inspect
import pandas as pd





db_pass = os.getenv('DB_PASS')

engine = create_engine(db_pass)
print(engine)



def obtain_data(security):

   # ssl_args = {'ssl': {'ca': 'ca-certificate.crt'}}
    engine = create_engine(db_pass)
    # Establish a connection
    with engine.connect() as connection:
        # Execute the query and load the result into a DataFrame
        df = pd.read_sql_query("""SELECT EURUSD.Date, EURUSD.Close as EURUSD_Close,EURUSD.Volume as EURUSD_Volume
                FROM EURUSD;""", connection)

    # Convert the 'date' column to datetime and then to Unix timestamp
    #df['Date'] = pd.to_datetime(df['Date'])
    #df['Date'] = df['Date'].apply(lambda x: x.timestamp())

    # Rename columns
   # df.rename(columns={ 'open': 'Open','high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]

    print(df)

    # Save the DataFrame to a CSV file
    df.to_csv(f'updated_data/train_data/pulled/{security}_joined.csv')



if __name__ in "__main__":
  
    security ='EURUSD'
    
    
    #for i in security:

    print(f'pulling {security}...')
       # if not table_exists(i, engine):
        #    add_table(i)
       # drop_table(i)
    obtain_data(security)