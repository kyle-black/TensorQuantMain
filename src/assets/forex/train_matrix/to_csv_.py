import pandas as pd
import os
import json





def csv_maker(security):

    with open(f'updated_data/train_data/redo/{security}.json', 'r') as f:
        data = [json.loads(line) for line in f]
    

    df= pd.DataFrame(data)
    df.rename(columns={'date': 'Date', 'open': 'Open','high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
    df.set_index('Date', inplace=True)

    df = df[~df.index.duplicated(keep='first')]
    df.to_csv(f'updated_data/train_data/redo/{security}.csv')
    
    '''
    df = pd.read_json(f'updated_data/train_data/redo/{security}.json')


    #df['date'] = pd.to_datetime(df['date'])
    #df['date'] = df['date'].apply(lambda x: x.timestamp())
    # Rename columns
    df.rename(columns={'date': 'Date', 'open': 'Open','high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)

    #print(df)
    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    df = df[~df.index.duplicated(keep='first')]

    print(df)


    #df = pd.read_json('updated_data/EURUSD.json')
    df.to_csv(f'updated_data/train_data/redo/csv/{security}.csv')
    '''



if __name__ == "__main__":

    path = 'updated_data/train_data/redo/'

    dir_list = os.listdir(path) 

    for asset in dir_list:
        print('converting', asset)
        asset_name = asset.split('.')[0]
        csv_maker(asset_name)

