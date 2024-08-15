import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import log_loss
import crossvalidation
import joblib




class MonitorActivation(layers.Layer):
    def __init__(self, **kwargs):
        super(MonitorActivation, self).__init__(**kwargs)
        self.activation_values = []

    def call(self, inputs):
        self.activation_values.append(tf.reduce_mean(inputs))
        return tf.nn.relu(inputs)




def run_model(df, asset, lookback, learning_rate=0.001, batch_size=256, epochs=1000):
    if asset is not None:
        asset = asset

    start_date = pd.to_datetime('2010-01-01')
    end_date = pd.to_datetime('2023-01-01')
    threshold = 0.7
    startlookback = lookback * 10

    df.dropna(inplace=True)
    df.dropna(how='all', inplace=True)

    df = df[startlookback:]
    df['endbarrier_unix'] = pd.to_datetime(df['endbarrier_unix'], unit='s')
    prices = df[['Close', 'touch_price', 'Date', 'endbarrier_unix', 'upper_barrier', 'lower_barrier', 'pct','prices_in_range']]
    df = df[['label', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI', 'Volume', '%K',
       '%D', 'daily_return', 'direction', 'volume_direction', 'OBV',
       'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
       'chikou_span', 'Close_AUDUSD', 'Close_USDCAD',
       'Close_USDCHF', 'AUDUSD_Returns', 'USDCAD_Returns', 'USDCHF_Returns']]
    
    #df['label'] = df['label'].map({-1: 0, 0: 0, 1: 1})

    train_datasets, test_datasets = crossvalidation.run_split_process(df)
    feature_cols = df.drop('label', axis=1).columns
    target_col = 'label'

    n_components = 20
    scaler = StandardScaler()

    train_idx = train_datasets[-3]
    test_idx = test_datasets[-3]

    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]
  #  prices = df[['Close', 'touch_price', 'Date', 'endbarrier_unix', 'upper_barrier', 'lower_barrier', 'pct','prices_in_range']]
    
    pct_change = prices['pct'].iloc[test_idx]
    startprice = prices['Close'].iloc[test_idx]
    endprice = prices['touch_price'].iloc[test_idx]
    Dates = prices['Date'].iloc[test_idx]
    enddate = prices['endbarrier_unix'].iloc[test_idx]
    upperbarrier = prices['upper_barrier'].iloc[test_idx]
    lowerbarrier = prices['lower_barrier'].iloc[test_idx]
    pricetouch = prices['prices_in_range'].iloc[test_idx]


    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(n_components,), kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(2, activation='softmax'))

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
    class_weights = dict(enumerate(class_weights))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping], class_weight=class_weights)
    #print("Mean activation values:", MonitorActivation.activation_values)
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')
    print(f'Test precision: {test_precision}')
    print(f'Test recall: {test_recall}')
    print(f'Test AUC: {test_auc}')

    y_pred_proba = model.predict(X_test)
    log_loss_value = log_loss(y_test, y_pred_proba)
    print(f'Log Loss: {log_loss_value}')

    proba_df = pd.DataFrame(y_pred_proba, columns=['Proba_Class_0', 'Proba_Class_1'])

    # Add back the selected columns to the test results
    test_results = test_data.reset_index(drop=True)
    test_results = pd.concat([test_results, proba_df], axis=1)
    test_results['True_Label'] = np.argmax(y_test, axis=1)

    # Adding back the columns from the prices DataFrame
    test_results['Close'] = startprice.reset_index(drop=True)
    test_results['touch_price'] = endprice.reset_index(drop=True)
    test_results['Date'] = Dates.reset_index(drop=True)
    test_results['endbarrier_unix'] = enddate.reset_index(drop=True)
    test_results['upper_barrier'] = upperbarrier.reset_index(drop=True)
    test_results['lower_barrier'] = lowerbarrier.reset_index(drop=True)
    test_results['price_in_range'] = pricetouch.reset_index(drop=True)

    test_results.to_csv('tester_df.csv')
    print(test_results)
    print('length of train data:',len(X_train))
    # Save the scaler
    joblib.dump(scaler, '../deploy/models/EURUSD/EURUSD_1024_4_scaler.pkl')

    # Save the PCA
    joblib.dump(pca, '../deploy/models/EURUSD/EURUSD_1024_4_pca.pkl')


    model.save('../deploy/models/EURUSD/EURUSD_1024_4.h5')



    return model, y_pred_proba, y_test, test_results