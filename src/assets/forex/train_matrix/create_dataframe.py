import pandas as pd
import os
import numpy as np
'''
# Directory containing the CSV files
directory = os.getcwd()
sub_dir = 'data/new_fx_data'
directory = os.path.join(directory, sub_dir)

csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Initialize a list to store the DataFrames
dfs = []

# Read each CSV file and append its DataFrame to the list
for file in csv_files:
    security_ = file.split('.csv')[0]
    file_path = os.path.join(directory, file)
    data = pd.read_csv(file_path)
    data.rename(columns={'Open':f'{security_}_Open', 'High': f'{security_}_High', 'Low':f'{security_}_Low', 'Close':f'{security_}_Close', 'Volume':f'{security_}_Volume', 'Volume MA':f'{security_}_Volume_MA'},inplace=True)
    data.set_index('Date', inplace=True)
    dfs.append(data)

# Concatenate all the DataFrames in the list into a single DataFrame
df = pd.concat(dfs, axis=1)
df.dropna(inplace=True)

df['datetime'] = pd.to_datetime(df.index, unit='s')

# Extract the day of the year
df['day_of_year'] = df['datetime'].dt.dayofyear

# Convert the day of the year to a value between -1 and 1 representing a sine wave
df['sine_date'] = np.sin(2*np.pi*df['day_of_year']/365.25)






df.to_csv('matrix.csv')

print(df)

'''

# Directory containing the CSV files
directory = os.getcwd()
sub_dir = 'updated_data/'
directory = os.path.join(directory, sub_dir)

json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

# Initialize a list to store the DataFrames
dfs = []

# Read each CSV file and append its DataFrame to the list
for file in json_files:
    security_ = file.split('.json')[0]
    file_path = os.path.join(directory, file)
    data = pd.read_json(file_path)
    data.rename(columns={'open':f'{security_}_Open', 'high': f'{security_}_High', 'low':f'{security_}_Low', 'close':f'{security_}_Close', 'volume':f'{security_}_Volume'},inplace=True)
    data.set_index('date', inplace=True)
    dfs.append(data)

# Concatenate all the DataFrames in the list into a single DataFrame
df = pd.concat(dfs, axis=1)
df.dropna(inplace=True)

df['datetime'] = pd.to_datetime(df.index, unit='s')

# Extract the day of the year
df['day_of_year'] = df['datetime'].dt.dayofyear

# Convert the day of the year to a value between -1 and 1 representing a sine wave
df['sine_date'] = np.sin(2*np.pi*df['day_of_year']/365.25)






df.to_csv('matrix.csv')

print(df)


