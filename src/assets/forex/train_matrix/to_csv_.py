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
    
   


if __name__ == "__main__":

    path = 'updated_data/train_data/redo/'

    dir_list = os.listdir(path) 

    for asset in dir_list:
        print('converting', asset)
        asset_name = asset.split('.')[0]
        if asset_name == 'NZDJPY':
            csv_maker(asset_name)

