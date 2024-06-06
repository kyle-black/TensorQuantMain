from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np
import crossvalidation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def ensemble_methods(df, asset,lookback):

    df = df.drop(columns=['touch_lower', 'touch_upper'])
    df = df.dropna(how='all')
    df = df[lookback:]


    print('input dataframe:',df.columns)
    
    print('splitting data ...')
    train_datasets, test_datasets, weights = crossvalidation.run_split_process(df)
   
    
    
    #feature_cols = ['Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI', 'SpreadOC', 'SpreadLH', 'SMI']
    df =df.drop(['Date','Close','Volume', 'upper_barrier', 'lower_barrier', 'pct_change', 't1'], axis =1)

    print('data dropped')
    
    feature_cols = df.drop('label',axis=1).columns
    

    target_col = 'label'

    print('featurecols:',feature_cols)
    

    all_predictions = []
    all_actuals = []
    all_preds = []
    n_components = 15
    scaler = StandardScaler()


    train_data =   train_datasets[-1]

    print('train_dataset', len(train_data))

    test_data = test_datasets[-1]

  

    weight_data =  weights[-1]
        
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]


    # Standardize the data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print('data scaled ...')
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)

   # X_train.to_csv('X_train.csv')
    
    X_test = pca.transform(X_test)

    # Initialize GridSearchCV
    #clf = SVC(probability=True, C=50)
    clf =RandomForestClassifier(n_jobs =-1,random_state=44, n_estimators=1000,criterion='gini')
    
    
    
    
   # clf = HistGradientBoostingClassifier()

    print('fitting model ...')
    clf.fit(X_train, y_train)
    print('model fitted ...')

    