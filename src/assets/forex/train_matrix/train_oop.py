import pandas as pd
import numpy as np
import bar_creation as bc
from statistics import variance
import numpy as np
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import probplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
import barriers
import features
from train_models import random_forest_classifier #, support_vector_classifier #, adaboost_classifier, random_forest_ts #, random_forest_anomaly_detector
from weights import return_attribution
#from train_models import neural_network_classifier

from autocorrelation import compute_and_plot_acf
import elbow_plot
from pca_maker import pca_



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
        return bc.get_volume_bars(self.time_bar_df, 10000)
    

    def dollar_bars(self):
        # Check if time_bar_df has been created, if not, create it
        if self.time_bar_df is None:
            self.time_bars()
        return bc.get_dollar_bars(self.time_bar_df, 7000)
    
    
class Analysis:
    def __init__(self, bars_df):
        # Store the bars dataframe regardless of its type (time, volume, dollar)
        self.bars_df = bars_df

    def std_dev(self):
        # Ensure bars_df is a DataFrame and has a 'Close' column
        if isinstance(self.bars_df, pd.DataFrame) and 'Daily_Returns' in self.bars_df.columns:
            std_dev_value = np.std(self.bars_df['Daily_Returns'][1:])
            return std_dev_value
        else:
            raise ValueError("Provided data is not a valid DataFrame or doesn't have a 'Daily Returns' column.")
        
    def jaque_bera(self):   # Test for normality
        jb_stat, p_value, _, _ = jarque_bera(self.bars_df['Daily_Returns'][1:])
        return jb_stat, p_value, _, _ 
    def AD_fuller(self): # Check for Stationary
        result = adfuller(self.bars_df['Daily_Returns'][1:])
        return  result
    def acf(self):
        acf_val = compute_and_plot_acf(self.bars_df['Close'][1:])

        return acf_val

class FeatureMaker:
    def __init__(self, bars_df, window, asset):
        # Store the bars dataframe regardless of its type (time, volume, dollar)
        self.bars_df = bars_df
        self.window =window
        self.asset= asset

    def feature_add(self):

        results =features.add_price_features(self.bars_df, self.window)

        return results

    def fac_diff(self):
        f_results = self.feature_add()
       # df =features.fractional_diff(f_results)
        d_values = features.find_min_d_for_df(f_results)
        return d_values
    
    def elbow_(self):
        result = self.feature_add()
        return elbow_plot.plot_pca(result)

    




class Labeling:
    def __init__(self, bars_df, asset):
        self.bars_df = bars_df
        self.asset =asset        

    def triple_barriers(self):
        self.triple_result =barriers.apply_triple_barrier(self.bars_df,[1,1,1], 72, self.asset)
        return self.triple_result
    
    def sample_weights(self):
        self.triple_result = self.triple_barriers()
        weights = return_attribution(self.triple_result)
        return weights
    
    


    


    

class Model:
    def __init__(self, bars_df, asset):
        self.bars_df = bars_df
        self.bar_shape = bars_df.shape
        self.asset = asset
        #self.weights =weights

    def train_model(self):
        #output =adaboost_classifier(self.bars_df)
        #output = support_vector_classifier(self.bars_df)
       # output =neural_network_cnn(self.bars_df, self.asset)
        output =random_forest_classifier(self.bars_df, self.asset, lookback =72)
        #output = neural_network_classifier(self.bars_df,self.asset)
        #output =random_forest_anomaly_detector(self.bars_df)
        return output
    
   




            
            
        
        



    




if __name__ == "__main__":
    
    asset ='EURUSD_60'

    lookback =72
    
    stock = pd.read_csv('updated_data/EURUSD.csv')


    stock.dropna(inplace=True)
    stock = stock.loc[~stock.index.duplicated(keep='first')]


   # print(stock.info(memory_usage='deep'))
   
  
    
    
    
    bar_creator = CreateBars(stock, asset)
   
    
    
    time_bars_df = bar_creator.time_bars()

    print(time_bars_df.tail())

    time_bars_df.to_csv('checktimebars.csv')
    
    #volume_bars_df = bar_creator.vol_bars()
    #print(volume_bars_df)

    #volume_bars_df.to_csv('volume_bars.csv')
    dollar_bars_df = bar_creator.dollar_bars()

    
    print(dollar_bars_df)

    #dollar_bars_df.to_csv('dollar_bars.csv')

    print(dollar_bars_df.columns)

    dollar_bars_df.to_csv('dollar_bars.csv')
    
    
    feature_instance_ = FeatureMaker(dollar_bars_df, 72, asset)
    
    feature_bars =feature_instance_.feature_add()
    print(feature_bars.columns)


    analysis_df = Analysis(dollar_bars_df)

    '''
    analysis_df.std_dev()
    analysis_df.jaque_bera()
    analysis_df.AD_fuller()
    analysis_df.acf()
    '''

    print('std dev:',analysis_df.std_dev())
    print('jaque bera:',analysis_df.jaque_bera())
    print('AD Fuller:', analysis_df.AD_fuller())
    print('ACF:', analysis_df.acf())

    

    feature_bars.to_csv('feature_bars.csv')
    
   # feature_instance_.elbow_()
    
    
    label_instance_ =Labeling(feature_bars, asset)
    label_instance_ = label_instance_.triple_barriers()
   # label_instance_ = label_instance_.add_pca()
    
    label_instance_.to_csv('labelinstance.csv')
    print(label_instance_)
    print(label_instance_.columns)
    
    label_instance_ = pd.read_csv('labelinstance.csv')
    asset ='EURUSD_60'
    

    model =Model(label_instance_, asset)
    
    print(model.train_model())
    
    
    
    
    


    

   










    

    

