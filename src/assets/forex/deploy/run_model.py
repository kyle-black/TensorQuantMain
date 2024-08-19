import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import features
from dollar_bars import dollar_bar_creator as dbc
import live_data










def prepare_data( dollar_threshold, asset, window_length):
    df = live_data.latest_data()
    dollar_df = dbc(asset,df,dollar_threshold)
    feature_df = features.add_price_features(df,asset, window_length)

    return feature_df



def make_predictions(pca,scaler, model, dollar_threshold, asset, window_length):
    df = prepare_data(dollar_threshold, asset, window_length)

    prediction_df = df.dropna()


    ##### Fit scaler
    #scaled_df = scaler.transform(prediction_df)
    #pca_df = pca.transform(scaled_df)

    ##### Fit Model

    #prediction_probas = model.fit(pca_df)
    
    return prediction_df.columns
    '''
    df = df[['Middle_Band', 'Upper_Band', 'Lower_Band', 'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI', 'Close', 'Volume', '%K',
       '%D', 'daily_return', 'direction', 'volume_direction', 'OBV',
       'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
       'chikou_span']]
    

    scaled_df = scaler.transform(df)
    pca_df = pca.transform(scaled_df)

    predictions = model.predict(pca_df)

    return predictions
    '''
    

if __name__ in "__main__":
    #Load TF model
    model = tf.keras.models.load_model('models/EURUSD/saved_models/EURUSD_1024_9.h5')

    # Load scaler
    scaler = joblib.load('models/EURUSD/saved_models/EURUSD_1024_9_scaler.pkl')

    # Load PCA
    pca = joblib.load('models/EURUSD/saved_models/EURUSD_1024_9_pca.pkl')
    
    dollar_threshold =10000
    asset ='EURUSD'
    window_length = 10

    print(make_predictions(pca, scaler, model, dollar_threshold, asset, window_length))


