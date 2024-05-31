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
        self.bars_df = bars_df

    def make_plot(self):
        return cp(self.bars_df, 'log_pct_change')

    def jaque_bera(self):   # Test for normality
        jb_stat, p_value, _, _ = jarque_bera(self.bars_df['log_pct_change'][1:])
        return jb_stat, p_value, _, _ 
    
    def ks_test(self):
        # Standardize the data (mean 0, standard deviation 1)
        #n_ = len(self.bars_df['Close'][500000:])
        #standardized_returns = (self.bars_df['log_pct_change'][1:] - self.bars_df['log_pct_change'][1:].mean()) / self.bars_df['log_pct_change'][1:].std()
        standardized_returns = (self.bars_df['log_pct_change'][1:])  
        
       
        # Perform the KS test against a normal distribution
        ks_stat, p_value = kstest(standardized_returns, 'norm')

        return ks_stat, p_value
    def AD_fuller(self): # Check for Stationary
        result = adfuller(self.bars_df['log_pct_change'][1:])
        return  result
    def plot_histogram(self):
        """Plot a histogram of the 'Returns' data."""
        plt.hist(self.bars_df['log_pct_change'][1:], bins=50, edgecolor='black')
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

    def triple_barriers(self):
        self.triple_result =barriers.new_apply_triple_barrier(self.bars_df,[1,1,1], lookback, self.asset)
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
        output =ensemble_methods(self.bars_df, self.asset, lookback)
        #output= Hist_boosted(self.bars_df, self.asset, lookback)
        #output = neural_network_classifier(self.bars_df,self.asset)
        #output =random_forest_anomaly_detector(self.bars_df)
        return output


    


if __name__ in "__main__":
    
    asset = 'EURUSD'
    dollar_amount =50000

    lookback = 60
    
    raw= pd.read_csv(f'merged.csv')
    
   # print('raw:',raw)

    for i in ['AUDUSD','USDCAD','USDCHF']:
    
        raw[f'{i}_Returns'] = raw[f'Close_{asset}'].pct_change()
    print(raw)
    
    cb = CreateBars(asset,raw, dollar_amount)
    df =cb.create_dollar_bars()
    print(df)
    
    #df.to_csv('dollarbar.csv')
    
    ad = Analysis(df)

    ad.create_plot()

    '''
    print(ad.jaque_bera())
    #print(df)
   # print(ad.ks_test())
    ad.plot_histogram()
   # ad.elbow_()
    print('adfuller:',ad.AD_fuller())
    '''
   
   
   
   
   
   
    '''
   
   
    #df = pd.read_csv('dollarbar.csv')

    #df['Returns'] = df[f'{asset}_Close'].pct_change()


    ad = Analysis(df)

    #print(ad.jaque_bera())
    print(df)
    print(ad.ks_test())
    ad.plot_histogram()
    print('adfuller:',ad.AD_fuller())
    

    print(df)
    L = Labeling(df,asset, lookback)
    df =L.triple_barriers()

    print('labeldf',df)
    df.to_csv('test_df.csv')

#    print('tEvents:',tEvents)

    
    fm = FeatureMaker(df, lookback, asset)


    df= fm.feature_add()
   # fm.elbow_()
    
    
    print('df test',df)

    
    
   # fm.elbow_()
    
    
   # cb = CreateBars(asset,raw, dollar_amount)
   # df =pd.read_csv('test_df.csv')
    lookback =20
    asset='EURUSD'
    fm = FeatureMaker(df, lookback, asset)
    
    tEvents = fm.create_CUMSUM_filter()
  

    # Filter df by tEvents
    filtered_df = df[df.index.isin(tEvents)]

   # filtered_df = df[df.isin(tEvents)]
    
    #filtered_df = filtered_df[42:]
    #filtered_df =df
    #filtered_df =df
    #filtered_df =df
    print('filtered_df:',filtered_df)
    filtered_df =filtered_df.dropna()
    #print('df:',filtered_df)
   # df = df.dropna()
    m = Model(filtered_df, asset)
    print(m.train_model())
    '''
    



    
  