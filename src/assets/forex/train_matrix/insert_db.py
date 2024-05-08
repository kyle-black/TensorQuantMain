import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float, DateTime, MetaData
from sqlalchemy import BigInteger
import os



db_pass = os.getenv('DB_PASS')



def add_table(security):
    # Read the CSV data
    df = pd.read_csv(f'updated_data/train_data/redo/{security}_R2.csv')

   # df['Datetime'] = pd.to_datetime(df['Date'], unit='s')

    
    df.insert(0, 'id', range(1, 1 + len(df)))
    
    # Set 'id' as the index
    df.set_index('id', inplace=True)

    ssl_args = {'ssl': {'ca': 'ca-certificate.crt'}}
    engine = create_engine(db_pass, connect_args=ssl_args)

    metadata = MetaData()

    
    table = Table(
        security, metadata,
        Column('id', Integer, primary_key=True),
        Column('Date', DateTime),
        Column('Open', Float),
        Column('Low', Float),
        Column('High', Float),
        Column('Close', Float),
        Column('Volume', Integer)
    )


    metadata.create_all(engine)

    df.to_sql(f'{security}', con=engine, if_exists='append', index_label='id')

def drop_table(security):
    ssl_args = {'ssl': {'ca': 'ca-certificate.crt'}}
    engine = create_engine(db_pass, connect_args=ssl_args)

    metadata = MetaData()

    # Reflect the tables
    metadata.reflect(bind=engine)

    # Drop the table
    Table(security, metadata, autoload_with=engine).drop(engine)


def obtain_data(security):

    ssl_args = {'ssl': {'ca': 'ca-certificate.crt'}}
    engine = create_engine(db_pass, connect_args=ssl_args)
    # Establish a connection
    with engine.connect() as connection:
        # Execute the query and load the result into a DataFrame
        df = pd.read_sql_query("""SELECT *
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
    df.to_csv(f'updated_data/train_data/pulled/{security}.csv')





if __name__ in "__main__":
    security = ['EURUSD']

    for i in security:
        #add_table(i)
        #drop_table(i)
        obtain_data(security)









