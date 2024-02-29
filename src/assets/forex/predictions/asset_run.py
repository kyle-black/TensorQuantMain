import prediction_oop
import time




def asset_(asset_n):
    prediction_oop.run_predictions(asset_n)
    #for asset in asset_list:
     #   prediction_oop.run_predictions(asset)
      #  print('Asset:', asset)
       # time.sleep(1)

if __name__ in "__main__":
    asset_name = 'AUDUSD'
    
    asset_(asset_name)