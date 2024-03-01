import pandas as pd
import numpy as np
import bar_creation as bc
import numpy as np
from redis_connect import url_connection



#from scipy.stats import norm

import barriers
import features
from train_models import random_forest_classifier #, adaboost_classifier, random_forest_ts #, random_forest_anomaly_detector
from weights import return_attribution

import elbow_plot
import prediction_fit
#import get_data
from datetime import datetime

import live_data

#from redis_connect import redis_connection
import time
import schedule
#import financial_bars  
from prediction_fit import make_predictions
import threading

class CreateBars:
    def __init__(self, raw_bars, asset):
        self.raw_bars = raw_bars
        self.time_bar_df = None  # Initialize this attribute
        self.time_bar_dict = None
        self.asset =asset
    def time_bars(self):
        self.time_bar_df = bc.time_bars(self.raw_bars, self.asset)  # Set the attribute
        self.time_bar_dict = self.time_bar_df.to_dict('records')
        return self.time_bar_df

    def vol_bars(self):
        # Check if time_bar_df has been created, if not, create it
        if self.time_bar_df is None:
            self.time_bars()
        return bc.get_volume_bars(self.time_bar_df, 10)
    

    def dollar_bars(self):
        # Check if time_bar_df has been created, if not, create it
        if self.time_bar_df is None:
            self.time_bars()
        return bc.get_dollar_bars_P(self.time_bar_df, 16000, self.asset)
    


class FeatureMaker:
    def __init__(self, bars_df, window):
        # Store the bars dataframe regardless of its type (time, volume, dollar)
        self.bars_df = bars_df
        self.window =window

    def feature_add(self):

        results =features.add_price_features(self.bars_df, self.window)

        return results

    def fac_diff(self):
        pass
    #    f_results = self.feature_add()
       # df =features.fractional_diff(f_results)
     #   d_values = features.find_min_d_for_df(f_results)
    #   return d_values
    
    def elbow_(self):
        pass
      
      #  result = self.feature_add()
       # return elbow_plot.plot_pca(result)

class Graphing:
    def __init__(self, bars_df):
        pass
        #self.bars_df =bars_df
        #self.figure = financial_bars.create_figure()

        #return self.figure.show()





class Labeling:
    def __init__(self, bars_df):
        self.bars_df = bars_df
        

    def triple_barriers(self):
        self.triple_result =barriers.apply_triple_barrier_P(self.bars_df,[1,1,1], 48)
        return self.triple_result
    
    def sample_weights(self):
        self.triple_result = self.triple_barriers()
        weights = return_attribution(self.triple_result)
        return weights
    


    

class Model:
    def __init__(self, symbol,bars_df):
        self.bars_df = bars_df

        self.symbol = symbol
        # predict on last 10 entriesail()
        self.bars_df = self.bars_df.tail(10)

        #self.weights =weights


    def predict_values(self):
        self.predictions, self.probas =make_predictions(self.symbol,self.bars_df)
       # self.predictions_dwn, self.probas_dwn =prediction_fit.make_predictions_dwn(self.symbol,self.bars_df)
        return self.predictions, self.probas, self.bars_df['upper_barrier'], self.bars_df['lower_barrier'], self.bars_df['Close'], self.bars_df['Open'], self.bars_df['High'], self.bars_df['Low']

    






def run_predictions(symbol):
    
    #stock = get_data.get_json(symbol)
    #symbol ='EURUSD'
    stock = live_data.latest_data_60(symbol)

   
    stock = stock.iloc[::]
    bar_creator = CreateBars(stock, symbol)
    
    
    
    dollar_bars_df = bar_creator.dollar_bars()
   
    print('Dollar bars:',dollar_bars_df)
    
    feature_instance_time = FeatureMaker(dollar_bars_df, 48)
    
    #print(dollar_bars_df)
    
    feature_bars =feature_instance_time.feature_add()
    #feature_instance_time.elbow_()

   

    
    feature_bars =feature_bars[['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
       'Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band',
       'Log_Returns', 'SpreadOC', 'SpreadLH', 'MACD', 'Signal_Line_MACD',
       'RSI','SMI']]
    
    
    label_instance_time =Labeling(feature_bars)
    label_instance_time = label_instance_time.triple_barriers()
    

   # print(label_instance_time)
 
    model =Model(symbol,label_instance_time)
    print(model.predict_values())

    output= model.predict_values()
    


    ################################# Parse Model output
    hard_predictions = output[0]
    last_hard_prediction = hard_predictions[-1]
    prob_predictions = output[1]
    dwn_prob = prob_predictions[-1][0]
    up_prob = prob_predictions[-1][2]
    neutral_prob = prob_predictions[-1][1]
    
    upper_barriers = output[2]
    lower_barriers = output[3]
    closes = output[4]
    opens = output[5]
    highs = output[6]
    lows = output[7]


    last_close = closes.index[-1]
    last_open = opens.index[-1]
    last_high = highs.index[-1]
    last_low = lows.index[-1]
    

    last_lower_barrier = round(lower_barriers.iloc[-1], 4)
    last_upper_barrier = round(upper_barriers.iloc[-1], 4)
    last_close = closes.iloc[-1]
    date = closes.index[-1]

    dt = closes.index[-1]

    unix_timestamp = int(dt.timestamp())

    #print('unix:', unix)
    
    #date = date.strftime('%Y-%m-%d %H:%M:%S')
    date_str = date.strftime('%Y-%m-%d %H:%M:%S')

    print(f'last close price:{last_close}\n last_upper_barrier: {last_upper_barrier} \n last_lower_barrier: {last_lower_barrier} \n predict_up: {up_prob} \n predict_down:{dwn_prob} \n neutral_prob:{neutral_prob} \n hard_prediction:{last_hard_prediction}' )

    output_dict = {}

    output_dict[date] = {'close': last_close, 'up_prob':up_prob,'dwn_prob':dwn_prob, 'neutral_prob':neutral_prob, 'upper_barrier':last_upper_barrier, 'lower_barrier':last_lower_barrier, 'hard_prediction':last_hard_prediction}

    print(output_dict)


    # Get the current time in Unix timestamp format
    current_unix_time = int(time.time())




    # Add the symbol to the Set
    url_connection.sadd('symbols', symbol)

    #unix_int = int(unix.timest)  

    
    # Add the timestamp to the Hash for the symbol
    #rl_connection.hset(symbol, unix_timestamp, f'{symbol}:{unix_timestamp}')

    # Add the data to the Hash for the symbol and timestamp
    
    url_connection.hset(
    f'{symbol}:{unix_timestamp}',
    mapping={
        'last_update_unix': current_unix_time,
        

        'security': symbol,
        'timeframe': '60m',
        'close': last_close,
        'up_prob': up_prob,
        'dwn_prob': dwn_prob,
        'neutral_prob': neutral_prob,
        'upper_barrier': last_upper_barrier,
        'lower_barrier': last_lower_barrier,
        'hard_prediction': last_hard_prediction,
        'time': date_str
        },
    )

   # url_connection.xadd(
   # f"security:{symbol}",
    #{'open':last_open,'high':last_high,'low':last_low,"close": last_close, "up_prob": up_prob, "dwn_prob": dwn_prob, "neutral_prob": neutral_prob,
    # 'upper_barrier': last_upper_barrier, 'lower_barrier':last_lower_barrier,'hard_prediction':last_hard_prediction,'time':date_str}
#)
   # last_open =0.0
    #last_high =0.0
    #last_low =0.0



    stream_name = f"security:{symbol}"
    entry = {"close": last_close, "up_prob": up_prob, "dwn_prob": dwn_prob, "neutral_prob": neutral_prob, 'upper_barrier': last_upper_barrier, 'lower_barrier':last_lower_barrier,'hard_prediction':last_hard_prediction,'time':date_str}



   # stream_name = f"security:{symbol}"
   # entry = {'open':last_open,'high':last_high,'low':last_low,"close": last_close, "up_prob": up_prob, "dwn_prob": dwn_prob, "neutral_prob": neutral_prob, 'upper_barrier': last_upper_barrier, 'lower_barrier':last_lower_barrier,'hard_prediction':last_hard_prediction,'time':date_str}

# Add entry to the stream
    url_connection.xadd(stream_name, entry)

# Retrieve and print all entries from the stream
    entries = url_connection.xrange(stream_name)
    for entry in entries:
        print('Redis update:',entry)


if __name__ in "__main__":
    
    
   
    
    print(run_predictions('USDCAD'))
    
    '''
    def run_assets():
       # symbols = ['AUDUSD','USDJPY', 'NZDUSD', 'USDCAD', 'USDCHF']
        symbols = [ 'USDCAD']
        threads = []

    # Create a new thread for each symbol
        for symbol in symbols:
            thread = threading.Thread(target=run_predictions, args=(symbol,))
            threads.append(thread)
            thread.start()

    # Wait for all threads to finish
        for thread in threads:
            thread.join()

    schedule.every(1).minutes.do(run_assets)

    while True:
        schedule.run_pending()
        time.sleep(1)
    '''