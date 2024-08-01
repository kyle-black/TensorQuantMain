from sklearn.model_selection import  GridSearchCV

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

import pandas as pd
# assuming crossvalidation and bootstrap are custom modules
import crossvalidation
#import bootstrap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import numpy as np
from sklearn.metrics import brier_score_loss
#import tensorflow as tf

import numpy as np


from sklearn.metrics import classification_report, confusion_matrix

#from sklearn.externals import joblib
# Import necessary keras modules

from sklearn.metrics import log_loss
from sklearn.dummy import DummyClassifier

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.utils import class_weight
import xgboost as xgb


import os 
import random


  

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.dummy import DummyClassifier






def ensemble_methods(df, asset, lookback):
    if asset is not None:
        asset = asset

    start_date = pd.to_datetime('2010-01-01')
    end_date = pd.to_datetime('2023-01-01')
    threshold = 0.7 
    df = df[lookback:]
    df['endbarrier_unix'] = pd.to_datetime(df['endbarrier_unix'], unit='s')

    prices = df[['Close', 'touch_price', 'Date', 'endbarrier_unix', 'upper_barrier', 'lower_barrier', 'pct']]
    df.drop(columns=['Date', 'unix', 'endbarrier_unix', 'Volume', 'Close', 'upper_barrier', 'lower_barrier', 'pct_change', 'Datetime', 'touch_price', 'prices_in_range', 'pct', 'Datehold'], inplace=True)
    df.dropna(how='all', inplace=True)
    df['label'] = df['label'].map({-1: 0, 0: 1, 1: 2})

    print('columns in training:', df.columns)
    train_datasets, test_datasets = crossvalidation.run_split_process(df)
    
    feature_cols = df.drop('label', axis=1).columns
    target_col = 'label'

    n_components = 15
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

    # Initialize XGBoost classifier
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    param_grid = {
        'n_estimators': [200],
        'max_depth': [3],
        'learning_rate': [0.1],
        'subsample': [ 0.9],
        'colsample_bytree': [ 0.9]
    }

    grid_search_xgb = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search_xgb.fit(X_train, y_train)
    best_xgb_clf = grid_search_xgb.best_estimator_

    best_xgb_clf.fit(X_train, y_train)

    dum = DummyClassifier(strategy='stratified', random_state=0)
    dum.fit(X_train, y_train)
    dum_score = dum.score(X_test, y_test)
    print('Dumb Score:', dum_score)

    real_score = best_xgb_clf.score(X_test, y_test)
    print('Real Score:', real_score)

    probas = best_xgb_clf.predict_proba(X_test)
    max_proba_indices = np.argmax(probas, axis=1)
    predicted_classes = best_xgb_clf.classes_[max_proba_indices]
    y_pred = predicted_classes

    print('######################')
    print('probas:', probas)
    print(classification_report(y_test, y_pred, zero_division=1))
    print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
    
    print(f'Y_true:{y_test} Y_pred:{y_pred}')

    comparison_df = pd.DataFrame({'Y_true': y_test, 'Y_pred': y_pred})
    print(comparison_df)

    all_actuals = y_test.tolist()
    all_preds = y_pred.tolist()
    print('###########################')

    classes = np.unique(y_train)
    l_l = log_loss(y_test, probas, labels=classes)
    print('logloss', l_l)  

    actual_ = []
    prediction_ = []
    dwn_ = []
    neutral_ = []
    up_ = []
    start_ = []
    end_ = []
    upper_ = []
    lower_ = []
    date_ = []
    enddate_ = []
    pct_change_ = []

    for actual, prediction, dwn, neutral, up, start, end, upperbarrier, lowerbarrier, date, end_date, pct_c in zip(y_test, y_pred, probas[:,0], probas[:,1], probas[:,2], startprice, endprice, upperbarrier, lowerbarrier, Dates, enddate, pct_change):
        actual_.append(actual)
        prediction_.append(prediction)
        dwn_.append(dwn)
        neutral_.append(neutral)
        up_.append(up)
        start_.append(start)
        end_.append(end)
        upper_.append(upperbarrier)
        lower_.append(lowerbarrier)
        date_.append(date)
        enddate_.append(end_date)
        pct_change_.append(pct_c)
        
    predictions_df2 = pd.DataFrame({
        'Actual': actual_,
        'Predictions': prediction_,
        'down proba': dwn_,
        'neutral proba': neutral_,
        'up proba': up_,
        'start': start_,
        'end': end_,
        'upper': upper_,
        'lower': lower_,
        'Dates': date_,
        'Endate': enddate_,
        'pct_change': pct_change_
    })

    predictions_df2.to_csv('predictions_df.csv', index=False)

    joblib.dump(best_xgb_clf, 'xgb_model.pkl')
    joblib.dump(pca, 'pca.pkl')
    joblib.dump(scaler, 'scaler.pkl')

# Example usage
# df = pd.read_csv('your_data.csv')
# ensemble_methods





    '''
    # Initialize an empty list to store the data
    data = []

# Iterate through the zipped lists

    print(type(startprice), type(endprice), type(upperbarrier), type(lowerbarrier), type(Dates), type(enddate))
    upperbarrier_series = pd.Series(upperbarrier)
    lowerbarrier_series = pd.Series(lowerbarrier)


 

# Optionally, inspect the first few rows to confirm data looks correct
    print('predictions:',prediction2_df.head())

   # print(test_data)
    test_data['predictions'] = y_pred
    test_data['probs dwn'] = probas[:,0]
    test_data['probs neutral'] = probas[:,1]
    test_data['probs up'] = probas[:,2]

    comparison_df.to_csv('comparison_df.csv')
   # prediction2_df.to_csv('predictions_df.csv')

    '''

def Hist_boosted(df, asset, lookback):
    if asset is not None:
        asset = asset

    # Data Preprocessing
    start_date = pd.to_datetime('2010-01-01')
    end_date = pd.to_datetime('2023-01-01')
    threshold = 0.7 

    df = df.drop(columns=['touch_lower', 'touch_upper','Volume','Close'])
    df = df.dropna(how='all')
    df = df[lookback:]

    print('input dataframe:',df.columns)

    df = df.drop(['Date', 'upper_barrier', 'lower_barrier','pct_change'], axis =1)

    X = df.drop('label',axis=1)
    y = df['label']

    # Initialize the classifiers with PCA and StandardScaler
    clf_hist = make_pipeline(StandardScaler(), PCA(n_components=10), HistGradientBoostingClassifier())
    clf_rf = make_pipeline(StandardScaler(), PCA(n_components=10), RandomForestClassifier())

    # Define parameter grid for HistGradientBoostingClassifier
    param_grid = {
    'histgradientboostingclassifier__max_iter': [10],
    'histgradientboostingclassifier__learning_rate': [ 0.1],
    'histgradientboostingclassifier__max_depth': [5],
    'histgradientboostingclassifier__min_samples_leaf': [50],
    'histgradientboostingclassifier__l2_regularization': [0.1],
    'histgradientboostingclassifier__max_bins': [255],
    'histgradientboostingclassifier__max_leaf_nodes': [None]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(clf_hist, param_grid, cv=4)

    # Perform grid search for Histogram Gradient Boosting
    grid_search.fit(X, y)
    print("Best parameters for Histogram Gradient Boosting: ", grid_search.best_params_)
    print("Best score for Histogram Gradient Boosting: ", grid_search.best_score_)

    scores = cross_validate(grid_search, X, y, cv=3,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)
    print(scores['test_neg_mean_squared_error'])
    


    # Perform 4-fold cross validation for Random Forest
    scores_rf = cross_val_score(clf_rf, X, y, cv=4)
    print("Random Forest cross-validation scores: ", scores_rf)
    print("Average Random Forest cross-validation score: ", scores_rf.mean())



# Fit the RandomForestClassifier
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X, y)

# Select one of the trees
   

    # Plot the tree
        # Export the tree to a .dot file
    