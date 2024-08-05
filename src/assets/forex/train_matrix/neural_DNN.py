import tensorflow as tf
from tensorflow.keras import layers, models

import pandas as pd
# assuming crossvalidation and bootstrap are custom modules
import crossvalidation
#import bootstrap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np



def run_model(df, asset, lookback):
    if asset is not None:
        asset = asset
   
    start_date = pd.to_datetime('2010-01-01')
    end_date = pd.to_datetime('2023-01-01')
    threshold = 0.7 
    
    startlookback = lookback * 10
    
    df = df[startlookback:]
    df['endbarrier_unix'] = pd.to_datetime(df['endbarrier_unix'], unit='s')

    prices = df[['Close', 'touch_price', 'Date', 'endbarrier_unix', 'upper_barrier', 'lower_barrier', 'pct']]
    df.drop(columns=['Date', 'unix', 'endbarrier_unix', 'Volume', 'Close', 'upper_barrier', 'lower_barrier', 'pct_change', 'Datetime', 'touch_price', 'prices_in_range', 'pct', 'Datehold', 'pips','change'], inplace=True)
    df.dropna(how='all', inplace=True)
    df['label'] = df['label'].map({-1: 0, 0: 1, 1: 2})

    train_datasets, test_datasets = crossvalidation.run_split_process(df)
    
    feature_cols = df.drop('label', axis=1).columns
    print('columns in training:', feature_cols)
    target_col = 'label'

    n_components = 15
    scaler = StandardScaler()

    train_idx = train_datasets[-5]
    test_idx = test_datasets[-5]

    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]
    pct_change = prices['pct'].iloc[test_idx]
    startprice = prices['Close'].iloc[test_idx]
    endprice = prices['touch_price'].iloc[test_idx]
    Dates = prices['Date'].iloc[test_idx]
    enddate = prices['endbarrier_unix'].iloc[test_idx]
    upperbarrier = prices['upper_barrier'].iloc[test_idx]
    lowerbarrier = prices['lower_barrier'].iloc[test_idx]

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Convert labels to categorical one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

    # Build the model
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(n_components,)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(3, activation='softmax'))

    # Compile the model with the categorical cross-entropy loss
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    # Evaluate the model
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')
    print(f'Test precision: {test_precision}')
    print(f'Test recall: {test_recall}')
    print(f'Test AUC: {test_auc}')

    # Get the predicted probabilities for the test set
    y_pred_proba = model.predict(X_test)

    # Return the predicted probabilities and true labels
    return model, y_pred_proba, y_test