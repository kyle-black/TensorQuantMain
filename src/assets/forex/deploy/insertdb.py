import pandas as pd
from sqlalchemy import create_engine

def insert_dataframe_to_mysql(df, table_name):
    # Create SQLAlchemy engine to connect to MySQL Database
    engine = create_engine('mysql+mysqlconnector://doadmin:AVNS_oW0kYA-LJsBz5pksVi4@tq-training-data-do-user-13042543-0.c.db.ondigitalocean.com:25060/defaultdb')

    # Insert the DataFrame into the MySQL database
    df.to_sql(name=table_name, con=engine, if_exists='append', index=False)

    print(f"DataFrame inserted into {table_name} table successfully.")

# Example DataFrame (assuming you have loaded your data into a DataFrame named 'df')
# df = pd.read_json('prediction/prediction_df.json')

# Insert the DataFrame into the 'EURUSD_Live' table
#insert_dataframe_to_mysql(df, 'EURUSD_Live')