import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float, DateTime, MetaData, VARCHAR
from sqlalchemy import BigInteger
import os
from sqlalchemy import inspect



db_pass = os.getenv('DB_PASS')


def table_exists(name, engine):
    inspector = inspect(engine)
    return name in inspector.get_table_names()



def add_table(security):
    # Read the CSV data
    df = pd.read_csv(f'updated_data/train_data/redo/{security}_R2.csv')

   # df['Datetime'] = pd.to_datetime(df['Date'], unit='s')

    
  #  df.insert(0, 'id', range(1, 1 + len(df)))
    df['Asset'] = security
    df['Asset_Type'] = 'forex'
    # Set 'id' as the index
    df.set_index('Date', inplace=True)

   # ssl_args = {'ssl': {'ca': 'ca-certificate.crt'}}
    engine = create_engine(db_pass)

    metadata = MetaData()

    
    table = Table(
        security, metadata,
      #  Column('id', Integer, primary_key=True),
        Column('Date', DateTime, primary_key=True),
        Column('Open', Float),
        Column('Low', Float),
        Column('High', Float),
        Column('Close', Float),
        Column('Volume', Integer),
        Column('Asset', VARCHAR(255)),
        Column('Asset_Type', VARCHAR(255)))


    metadata.create_all(engine)

    df.to_sql(f'{security}', con=engine, if_exists='append', index_label='Date')

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
        df = pd.read_sql_query("""SELECT EURUSD.Date, EURUSD.Close as EURUSD_Close,EURUSD.Volume as EURUSD_Volume, AUDUSD.Close as AUDUSD_Close, 
                USDCAD.Close as USDCAD_Close,
                USDCHF.Close as USDCHF_Close,
                USDHKD.Close as USDHKD_Close,
                USDJPY.Close as USDJPY_Close
                FROM EURUSD 
                JOIN AUDUSD ON AUDUSD.Date = EURUSD.Date    
                JOIN USDCAD ON USDCAD.Date = EURUSD.Date
                JOIN USDCHF ON USDCHF.Date = EURUSD.Date
                JOIN USDHKD ON USDHKD.Date = EURUSD.Date
                JOIN USDJPY ON USDJPY.Date = EURUSD.Date;""", connection)

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
  #  db_pass = os.getenv('DB_PASS')

   # ssl_args = {'ssl_ca':'ca-certificate.crt'}
   # engine = create_engine(db_pass)
    
   # security = ['EURUSD','USDCAD','AUDUSD','NZDJPY','GBPJPY','USDCHF','USDHKD','USDJPY']
    security ='EURUSD'
    
    
    #for i in security:

    print(f'pulling {security}...')
       # if not table_exists(i, engine):
        #    add_table(i)
       # drop_table(i)
    obtain_data(security)









