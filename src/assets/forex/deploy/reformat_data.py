import pandas as pd
import numpy as np







training_cols = ['label', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI', 'Volume', '%K',
       '%D', 'daily_return', 'direction', 'volume_direction', 'OBV',
       'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
       'chikou_span', 'Close_AUDUSD', 'Close','volatility','day_of_week']




def format(df, security):


    df.rename{colomns = ''}