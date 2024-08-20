import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import features
from dollar_bars import dollar_bar_creator as dbc
import live_data
import barriers










def prepare_data( dollar_threshold, asset, window_length, securities):
    df = live_data.latest_data(securities)
    dollar_df = dbc(asset,df,dollar_threshold)
  #  barrier_df =barriers.calculate_barriers_R(dollar_df)
    feature_df = features.add_price_features(dollar_df,asset, window_length)

    return feature_df



def make_predictions(pca,scaler, model, dollar_threshold, asset, window_length, securities):
    df = prepare_data(dollar_threshold, asset, window_length,securities)
    print('pre prediction df:', df.columns)
   # prediction_df = df.dropna()

    prediction_df = df.drop(columns=['Date', 'Datehold', 'change', 'pct_change', 'pips'])

    prediction_df = prediction_df[['Middle_Band', 'Upper_Band', 'Lower_Band', 'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI', 'Volume', '%K',
        '%D', 'daily_return', 'direction', 'volume_direction', 'OBV',
       'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
       'chikou_span', 'Close_AUDUSD', 'Close','volatility','day_of_week']]
    print(prediction_df)
    ##### Fit scaler
    scaled_df = scaler.transform(prediction_df)
    pca_df = pca.transform(scaled_df)

    ##### Fit Model

    prediction_probas = model.fit(pca_df)
    
    return prediction_probas
    '''
    df = df[['Middle_Band', 'Upper_Band', 'Lower_Band', 'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI', 'Close', 'Volume', '%K',
       '%D', 'daily_return', 'direction', 'volume_direction', 'OBV',
       'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
       'chikou_span']]
    

    scaled_df = scaler.transform(df)python3 
    pca_df = pca.transform(scaled_df)

    predictions = model.predict(pca_df)

    return predictions
    '''
    

if __name__ in "__main__":
    #Load TF model
    model = tf.keras.models.load_model('models/EURUSD/EURUSD_0816-20.h5')

    # Load scaler
    scaler = joblib.load('models/EURUSD/EURUSD_0816-20_scaler.pkl')

    # Load PCA
    pca = joblib.load('models/EURUSD/EURUSD_0816-20_pca.pkl')
    #pca =None
    #scaler =None
    #model =None
    dollar_threshold =10000
    asset ='EURUSD'
    window_length = 10
    securities = ['EURUSD', 'AUDUSD']

    print(make_predictions(pca, scaler, model, dollar_threshold, asset, window_length, securities))


