from dollar_bars import dollar_bar_creator as dbc

from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kstest
import matplotlib.pyplot as plt

import barriers
import features
import elbow_plot
from pca_maker import pca_
from weights import return_attribution
from CUMSUM_filter import gTEvents as gte
import pandas as pd
#from train_models import ensemble_methods #random_forest_classifier, Hist_boosted

from check_distro import create_plot as cp
import numpy as np
from scipy.stats import boxcox
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import dask.dataframe as dd
import send_email
import neural_DNN










class CreateBars:
    """Class to create dollar bars from asset data."""

    def __init__(self, asset,gRaw, amnt):
        """Initialize with asset data."""
        self.asset = asset
        self.dollar_bars = None
        self.dollar_amt = amnt
        self.raw_bars = gRaw
        self.tEvents = None

    

    def create_dollar_bars(self):
        """Create dollar bars from asset data."""
        self.dollar_bars = dbc(self.asset,self.raw_bars,self.dollar_amt )
        return self.dollar_bars
    
    def create_CUMSUM_filter(self, bars):
        """Create dollar bars from asset data."""
        self.tEvents = gte(bars, None)
        return self.tEvents


class Analysis:
    def __init__(self, bars_df):
        # Store the bars dataframe regardless of its type (time, volume, dollar)

        bars_df['log_return'] = np.log(bars_df['pips'] / bars_df['pips'].shift(1)).dropna()
        min_log_return = bars_df['log_return'].min()
        
        if min_log_return < 0:
            bars_df['log_return'] = bars_df['log_return'] - min_log_return + 1  # Add 1 to avoid zero values

# Apply Box-Cox transformation
        bars_df = bars_df.dropna(subset=['log_return'])
        bars_df['transformed_log_return'], lambda_value = boxcox(bars_df['log_return'].dropna())
        bars_df = bars_df.dropna()
      #  bars_df.loc[:, 'detrended_boxcox_price'] = bars_df['boxcox_price'] - bars_df['boxcox_price'].rolling(window=12).mean()
        bars_df = bars_df.dropna()
        
       # bars_df.loc[:, 'diff_boxcox_price'] = bars_df['boxcox_price'].diff()
        bars_df = bars_df.dropna()
        self.bars_df = bars_df

    def make_plot(self):
        return cp(self.bars_df, 'pct_change')

    def jaque_bera(self):   # Test for normality
        jb_stat, p_value, _, _ = jarque_bera(self.bars_df['transformed_log_return'][2:])
        return jb_stat, p_value, _, _ 
    
    def ks_test(self):
        # Standardize the data (mean 0, standard deviation 1)
        #n_ = len(self.bars_df['Close'][500000:])
        standardized_returns = (self.bars_df['transformed_log_return'][2:] - self.bars_df['transformed_log_return'][2:].mean()) / self.bars_df['transformed_log_return'][2:].std()
      #  standardized_returns = (self.bars_df['boxcox_price'][2:])  
        
       
        # Perform the KS test against a normal distribution
        ks_stat, p_value = kstest(standardized_returns, 'norm')

        return ks_stat, p_value
    def AD_fuller(self): # Check for Stationary
        result = adfuller(self.bars_df['transformed_log_return'][2:])
        return  result
    def plot_histogram(self):
        """Plot a histogram of the 'Returns' data."""
        plt.hist(self.bars_df['transformed_log_return'][2:], bins=50, edgecolor='black')
        plt.title('Histogram of Returns')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.savefig('histogram.png')



class FeatureMaker:
    def __init__(self, bars_df, window, asset):
        # Store the bars dataframe regardless of its type (time, volume, dollar)
        self.bars_df = bars_df
        self.window =window
        self.asset= asset
        

    def feature_add(self):

        results =features.add_price_features(self.bars_df, self.asset, self.window)

        return results

    def fac_diff(self):
        f_results = self.feature_add()
       # df =features.fractional_diff(f_results)
        d_values = features.find_min_d_for_df(f_results)
        return d_values
    
    def elbow_(self):
        result = self.feature_add()
        return elbow_plot.plot_pca(result)
    
    def create_CUMSUM_filter(self):
        
        pass       
        """Create dollar bars from asset data."""
#        self.tEvents = gte(self.bars_df, self.dv)
#        return self.tEvents

    




class Labeling:
    def __init__(self, bars_df, asset,lookback):
        self.bars_df = bars_df
        self.asset =asset
        self.lookback =lookback
        self.bars_df.index = pd.to_datetime(self.bars_df.Date)
        self.daily_volatility = self.bars_df['Close'].pct_change().std()

    

        
    def triple_barriers(self):
        self.triple_result =barriers.calculate_barriers_R(self.bars_df, self.lookback)
      
        self.bars_df = self.triple_result

      
        return self.bars_df
    
    def sample_weights(self):
        self.triple_result = self.triple_barriers()
        weights = return_attribution(self.triple_result)
        return weights
    
    


    


    

class Model:
    def __init__(self, bars_df, asset,lookback,n_components,training_cols,model_num, learning_rate,batch_size,epochs):
        self.bars_df = bars_df
        self.bar_shape = bars_df.shape
        self.asset = asset
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_components =n_components
        self.training_cols =training_cols
        self.model_num = model_num
        #self.weights =weights

    def train_model(self):
      
        output =neural_DNN.run_model(self.bars_df, self.asset,self.lookback,self.n_components,self.training_cols,self.model_num, self.learning_rate,self.batch_size, self.epochs)
        
        
        return output


def prepare_data(dollar_amount, lookback):
   
    
    # Use Dask to read the CSV file in chunks
    raw = dd.read_csv('merged.csv')
    raw = raw[:]

    # Compute the result and convert to a pandas DataFrame
    raw = raw.compute()
    raw = raw[:-50]
    print(raw)
    cb = CreateBars(asset,raw, dollar_amount)
    print('Creating Dollar Bars')
    df =cb.create_dollar_bars()

    # Save the DataFrame in a more efficient format
    df.to_parquet('inf_check.parquet')
    
   
    df = pd.read_parquet('inf_check.parquet')
    L = Labeling(df,asset, lookback)
    
    
    print('Applying Triple Barriers:')
    df = L.triple_barriers()
 
    fm = FeatureMaker(df, lookback, asset)
    df= fm.feature_add()
    

    
  

    filtered_df =df
    filtered_df.to_parquet('final_df.parquet')
   
    
    return filtered_df 
    
    
    
    
def train_data(df, asset,lookback,n_components,training_cols,model_num,learning_rate,batch_size,epochs):
    m = Model(df, asset,lookback,n_components,training_cols,model_num,learning_rate,batch_size,epochs)
    print(m.train_model())




if __name__ == "__main__":
    #hyperparameter for experiment and model creation
    model_num ='EURUSD_0816-8'
    asset = "EURUSD"
    dollar_amount =10000
    lookback =10
    n_components = 17
    learning_rate =.001
    batch_size =128
    epochs =200

    training_cols = ['label', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI', 'Volume', '%K',
       '%D', 'daily_return', 'direction', 'volume_direction', 'OBV',
       'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
       'chikou_span', 'Close_AUDUSD', 'Close','volatility']




    prepare_data(dollar_amount, lookback)
    
   # send_email.run_email()
    df = pd.read_parquet('final_df.parquet')
    train_data(df, asset,lookback,n_components,training_cols,model_num,learning_rate,batch_size,epochs)

    e_df = pd.DataFrame()

    e_df['model_num'] = model_num
    e_df['asset'] = asset
    e_df['dollar_amount'] = dollar_amount
    e_df['n_components'] = n_components
    e_df['lookback'] = lookback
    e_df['learning_rate'] = learning_rate
    e_df['batch_size'] = batch_size
    e_df['epochs']  =epochs
    e_df['training_cols'] =str(training_cols)
    
    e_df.to_csv('experiment_tracker.csv', index=False, header=False)
 
    send_email.run_email(model_num)
    
