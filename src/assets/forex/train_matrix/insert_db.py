import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float, DateTime, MetaData, VARCHAR
from sqlalchemy import BigInteger
import os
from sqlalchemy import inspect
import pymysql
pymysql.install_as_MySQLdb()



db_pass = os.getenv('DB_PASS')


def table_exists(name, engine):
    inspector = inspect(engine)
    return name in inspector.get_table_names()



def add_table(security):
    ssl_args = {'ssl': {'ca': 'ca-certificate.crt'}}
    engine = create_engine(db_pass, connect_args=ssl_args)
    # Read the CSV data
    df = pd.read_csv(f'updated_data/train_data/redo/{security}.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    df['Asset'] = security
    df['Asset_Type'] = 'forex'
    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    engine = create_engine(db_pass)

    df.to_sql(f'{security}', con=engine, if_exists='replace', index=True, index_label='Date')

def drop_table(security):
    ssl_args = {'ssl': {'ca': 'ca-certificate.crt'}}
    engine = create_engine(db_pass, connect_args=ssl_args)

    metadata = MetaData()

    # Reflect the tables
    metadata.reflect(bind=engine)

    # Drop the table
    Table(security, metadata, autoload_with=engine).drop(engine)


def obtain_data(security):

   # ssl_args = {'ssl': {'ca': 'ca-certificate.crt'}}
    engine = create_engine(db_pass)
    # Establish a connection
    with engine.connect() as connection:
        # Execute the query and load the result into a DataFrame
        df = pd.read_sql_query("""SELECT E.Date as Date, E.Close as EURUSD_CLOSE, C.Close as USDCAD_Close FROM EURUSD E JOIN USDCAD C ON C.Date =E.Date;""", connection)

    # Convert the 'date' column to datetime and then to Unix timestamp
    #df['Date'] = pd.to_datetime(df['Date'])
    #df['Date'] = df['Date'].apply(lambda x: x.timestamp())

    # Rename columns
   # df.rename(columns={ 'open': 'Open','high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)

    # Set 'Date' as the index
   # df.set_index('Date', inplace=True)

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]

    print(df)

    # Save the DataFrame to a CSV file
    df.to_csv(f'updated_data/train_data/pulled/{security}_joined.csv')





if __name__ in "__main__":
  #  db_pass = os.getenv('DB_PASS')

   # ssl_args = {'ssl_ca':'ca-certificate.crt'}
   # engine = create_engine(db_pass)
    
   # security = ['EURUSD','USDCAD','AUDUSD','NZDJPY','GBPJPY','USDCHF','USDHKD','USDJPY']
    security ='NZDJPY'
    
    
    #for i in security:

    print(f'pulling {security}...')
       # if not table_exists(i, engine):
    add_table(security)
    #drop_table(security)
    #obtain_data(security)









