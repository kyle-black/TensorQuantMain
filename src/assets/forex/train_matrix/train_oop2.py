from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller
from dollar_bars import dollar_bar_creator as dbc


class CreateBars:
    def __init__(self, asset, dollar_amt):
        
        self.asset =asset
        self.dollar_amt = dollar_amt
        self.dollar_bars =None


    def dollar_bars(self):
        # Check if time_bar_df has been created, if not, create it
        self.dollar_bars = dbc.get_dollar_bars(self.asset, self.dollar_amt)
        return self.dollar_bars
    

class Analysis:
    def __init__(self, bars_df):
        # Store the bars dataframe regardless of its type (time, volume, dollar)
        self.bars_df = bars_df

    def std_dev(self):
        # Ensure bars_df is a DataFrame and has a 'Close' column
        if isinstance(self.bars_df, pd.DataFrame) and 'Close' in self.bars_df.columns:
            std_dev_value = np.std(self.bars_df['Close'][1:])
            return std_dev_value
        else:
            raise ValueError("Provided data is not a valid DataFrame or doesn't have a 'Close' column.")
        
    def jaque_bera(self):   # Test for normality
        jb_stat, p_value, _, _ = jarque_bera(self.bars_df['Returns'][1:])
        return jb_stat, p_value, _, _ 
    def AD_fuller(self): # Check for Stationary
        result = adfuller(self.bars_df['Close'][1:])
        return  result
    def acf(self):
        pass




if __name__ in "__main__":

    cb = CreateBars('EURUSD', 10000)

    df = cb.dollar_bars()

    ay = Analysis(df)

    print(ay.jaque_bera())
