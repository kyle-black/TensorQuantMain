import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import features
import bar_creation as bc



#Load TF model
model = tf.keras.models.load_model('models/EURUSD/saved_models/EURUSD_1024_1.h5')

# Load scaler
scaler = joblib.load('models/EURUSD/saved_models/EURUSD_1024_1_scaler.pkl')

# Load PCA
pca = joblib.load('models/EURUSD/saved_models/EURUSD_1024_1_pca.pkl')





def get_live_data():
    pass


def prepare_data(df, dollar_threshold, asset, window_length):
    dollar_df = bc.get_dollar_bars(df,dollar_threshold)
    feature_df = features.add_price_features(df,asset, window_length)

    return feature_df



def make_predictions(pca,scaler, model, df, dollar_threshold, asset, window_length):
    df = prepare_data(df, dollar_threshold, asset, window_length)
    df = df[['Middle_Band', 'Upper_Band', 'Lower_Band', 'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI', 'Close', 'Volume', '%K',
       '%D', 'daily_return', 'direction', 'volume_direction', 'OBV',
       'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
       'chikou_span']]
    

    scaled_df = scaler.transform(df)
    pca_df = pca.transform(scaled_df)

    predictions = model.predict(pca_df)

    return predictions
    
    
    



