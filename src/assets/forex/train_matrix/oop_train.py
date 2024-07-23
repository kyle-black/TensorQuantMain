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
from train_models import ensemble_methods #random_forest_classifier, Hist_boosted

from check_distro import create_plot as cp
import numpy as np
from scipy.stats import boxcox
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import dask.dataframe as dd
import send_email









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
        """Create dollar bars from asset data."""
        self.tEvents = gte(self.bars_df, None)
        return self.tEvents

    




class Labeling:
    def __init__(self, bars_df, asset,lookback):
        self.bars_df = bars_df
        self.asset =asset
        self.lookback =lookback
        self.bars_df.index = pd.to_datetime(self.bars_df.Date)
        self.daily_volatility = self.bars_df['Close'].pct_change().std()

    

        
    def triple_barriers(self):
        self.triple_result =barriers.calculate_barriers_R(self.bars_df, self.lookback)
      #  print(self.triple_barriers.columns)
      #  print(self.barrierss)
        self.bars_df = self.triple_result

      #  print(self.bars_df.columns)
        #self.bars_df = pd.concat([self.bars_df, self.triple_result], axis=1)
       # self.triple_result = self.new_apply_triple_barrier(self.bars_df, [1,1,1], self.lookback, self.asset)
        return self.bars_df
    
    def sample_weights(self):
        self.triple_result = self.triple_barriers()
        weights = return_attribution(self.triple_result)
        return weights
    
    


    


    

class Model:
    def __init__(self, bars_df, asset,lookback):
        self.bars_df = bars_df
        self.bar_shape = bars_df.shape
        self.asset = asset
        self.lookback = lookback
        #self.weights =weights

    def train_model(self):
        #output =adaboost_classifier(self.bars_df)
        #output = support_vector_classifier(self.bars_df)
       # output =neural_network_cnn(self.bars_df, self.asset)
        output =ensemble_methods(self.bars_df, self.asset, self.lookback)
        #output= Hist_boosted(self.bars_df, self.asset, lookback)
        #output = neural_network_classifier(self.bars_df,self.asset)
        #output =random_forest_anomaly_detector(self.bars_df)
        return output


def prepare_data():
    
    asset = 'EURUSD'
    dollar_amount =1000
    lookback = 48
    
    # Use Dask to read the CSV file in chunks
    raw = dd.read_csv('merged.csv')
  #  raw = raw[:]

    # Compute the result and convert to a pandas DataFrame
    raw = raw.compute()
    raw = raw[:]
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
    print(df)
    
    
   # return df
    
   # df.to_parquet('barrier_df.parquet')
    #return df
    
   # df.to_parquet('test_df.parquet')
    #print('testdf:',df)
   # df = pd.read_parquet('barrier_df.parquet')
    fm = FeatureMaker(df, lookback, asset)
    df= fm.feature_add()

   # df.to_parquet('final_df.parquet')


    #df = pd.read_parquet('final_df.parquet')
   # fm = FeatureMaker(df, lookback, asset)
   # tEvents = fm.create_CUMSUM_filter()
   # filtered_df = df[df.index.isin(tEvents)]
    
    #print('filtered_df:',filtered_df)
    filtered_df = df
    #filtered_df =df
    filtered_df.dropna(inplace=True)
    
    return filtered_df
    
    
    
def train_data(filtered_df,asset,lookback):
    m = Model(filtered_df, asset,lookback)
    print(m.train_model())




if __name__ == "__main__":
    asset = "EURUSD"
    df = prepare_data()
    
   # send_email.run_email()

    train_data(df, asset,720)
    send_email.run_email()
    
