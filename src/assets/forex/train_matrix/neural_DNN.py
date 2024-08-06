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

from sklearn.metrics import log_loss

def run_model(df, asset, lookback):
    if asset is not None:
        asset = asset
   
    start_date = pd.to_datetime('2010-01-01')
    end_date = pd.to_datetime('2023-01-01')
    threshold = 0.7 
    
    startlookback = lookback * 10
    
    df = df[startlookback:]
    df['endbarrier_unix'] = pd.to_datetime(df['endbarrier_unix'], unit='s')
    print('columns pre:', df.columns)
    prices = df[['Close', 'touch_price', 'Date', 'endbarrier_unix', 'upper_barrier', 'lower_barrier', 'pct']]
    #df.drop(columns=[ 'Close_AUDUSD', 'Close_USDCAD','Close_USDCHF', 'AUDUSD_Returns', 'USDCAD_Returns', 'USDCHF_Returns','durableGoods', '15Yr_Fixed', '30Yr_Fixed', 'CPI', 'GDP','Production_Total_Index', 'Yields_COD', 'consumerSentiment','federalFunds', 'inflation', 'inflationRate', 'initialClaims','nominalPotentialGDP', 'rates_CreditCards', 'realGDP','realGDPPerCapita', 'retailMoneyFunds', 'retailSales', 'pips', 'change','pct_change', 'Datehold', 'day_of_week', 'Datetime', 'unix','upper_barrier', 'lower_barrier', 'endbarrier_unix', 'prices_in_range','touch_price', 'pct'], inplace=True)
    df = df[[ 'label','Close_AUDUSD', 'Close_USDCAD','Close_USDCHF','AUDUSD_Returns', 'USDCAD_Returns', 'USDCHF_Returns' ]]
    print('new_df',df.head())
    df.dropna(how='all', inplace=True)
    df['label'] = df['label'].map({-1: 0, 0: 1, 1: 2})

    train_datasets, test_datasets = crossvalidation.run_split_process(df)
    feature_cols = df.drop('label', axis=1).columns
    target_col = 'label'

    n_components = 3
    scaler = StandardScaler()

    train_idx = train_datasets[-1]
    test_idx = test_datasets[-1]

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

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(n_components,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[early_stopping])

    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')
    print(f'Test precision: {test_precision}')
    print(f'Test recall: {test_recall}')
    print(f'Test AUC: {test_auc}')

    y_pred_proba = model.predict(X_test)
    log_loss_value = log_loss(y_test, y_pred_proba)
    print(f'Log Loss: {log_loss_value}')

    # Convert predicted probabilities to a DataFrame
    proba_df = pd.DataFrame(y_pred_proba, columns=['Proba_Class_0', 'Proba_Class_1', 'Proba_Class_2'])

    # Add true labels and other relevant information from the test set
    test_results = test_data.reset_index(drop=True)
    test_results = pd.concat([test_results, proba_df], axis=1)
    test_results['True_Label'] = np.argmax(y_test, axis=1)
    test_results.to_csv('tester_df.csv')
    # Print the test DataFrame with probabilities
    print(test_results)

    return model, y_pred_proba, y_test, test_results