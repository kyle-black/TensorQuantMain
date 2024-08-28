import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import features
from dollar_bars import dollar_bar_creator as dbc
import live_data
import barriers










def prepare_data( dollar_threshold, asset, window_length, securities):
   #df = live_data.latest_data(securities)
    df = live_data.combine_data()

    dollar_df = dbc(asset,df,dollar_threshold)
  #  barrier_df =barriers.calculate_barriers_R(dollar_df)
    feature_df = features.add_price_features(dollar_df,asset, window_length)

    return feature_df



def make_predictions(pca, scaler, model, dollar_threshold, asset, window_length, securities):
    df = prepare_data(dollar_threshold, asset, window_length, securities)
    print('pre prediction df:', df)

    df.to_csv('prediction.csv')

    # Drop unnecessary columns
    prediction_df = df.drop(columns=['Date', 'Datehold', 'change', 'pct_change', 'pips'])
    print('predictiondf check', prediction_df) 
    
    # Drop rows with missing values
    prediction_df.dropna(inplace=True)
    
    # Select only relevant columns for prediction
    prediction_df = prediction_df[['Middle_Band', 'Upper_Band', 'Lower_Band', 'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI', 'Volume', '%K',
                                   '%D', 'daily_return', 'direction', 'volume_direction', 'OBV',
                                   'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
                                   'chikou_span', 'Close_AUDUSD', 'Close', 'volatility', 'day_of_week']]
    
    print('prescaled ####', prediction_df[-200:])
    
    # Use the last 200 rows for prediction
    prediction_df = prediction_df[-200:]
    
    # Scale the data
    scaled_df = scaler.transform(prediction_df)
    print(scaled_df)
    
    # Apply PCA transformation
    pca_df = pca.transform(scaled_df)
    
    # Make predictions using the model
    prediction_probas = model.predict(pca_df)
    negative_class_proba = prediction_probas[:, 0]
    positive_class_proba = prediction_probas[:, 1]
    # Add prediction probabilities to the DataFrame
    prediction_df['prediction_proba_dwn'] = negative_class_proba
    prediction_df['prediction_proba_up'] = positive_class_proba


    prediction_df.to_csv('prediction_df.csv')
    
    return prediction_df
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
    model = tf.keras.models.load_model('models/EURUSD/EURUSD_0816-18.h5')

    # Load scaler
    scaler = joblib.load('models/EURUSD/EURUSD_0816-18_scaler.pkl')

    # Load PCA
    pca = joblib.load('models/EURUSD/EURUSD_0816-18_pca.pkl')
   # pca =None
   # scaler =None
   # model =None
    dollar_threshold =10000
    asset ='EURUSD'
    window_length = 10
    securities = ['EURUSD', 'AUDUSD']

    print(make_predictions(pca, scaler, model, dollar_threshold, asset, window_length, securities))


