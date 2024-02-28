import prediction_oop
import time




def asset_(asset_list):

    for asset in asset_list:
        prediction_oop.run_predictions(asset)
        print('Asset:', asset)
        time.sleep(1)

if __name__ in "__main__":
    asset_list = ['EURUSD', 'AUDUSD']
    
    asset_(asset_list)