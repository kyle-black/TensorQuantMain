from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss
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
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import brier_score_loss
#from sklearn.externals import joblib
# Import necessary keras modules
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.dummy import DummyClassifier


#from sklearn.experimental import enable_hist_gradient_boosting  # noqa
#from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.utils import class_weight
#import xgboost as xgb
#from imblearn.over_sampling import SMOTE

#from sklearn.tree import export_graphviz
#import pydotplus

#import neuralnet
#from tensorflow.keras.utils import to_categorical


#from keras.models import Sequential
#from keras.layers import Dense, Dropout

#from keras.utils import to_categorical
#from keras.optimizers import Adam
#import tensorflow as tf
import os 
import random

#os.environ['PYTHONHASHSEED']=str(0)
#random.seed(0)
#np.random.seed(0)
#tf.random.set_seed(0)

    # You might want to return something from this function, like t
  

def ensemble_methods(df, asset, lookback):
    
    
    if asset is not None:
        asset =asset
    

    # Data Preprocessing
    start_date = pd.to_datetime('2010-01-01')
    end_date = pd.to_datetime('2023-01-01')
    threshold = 0.7 


    
    #df = df.drop(columns=['touch_lower', 'touch_upper'])
    #df = df.dropna(how='all')
    #df = df[lookback:]
    # Drop unnecessary columns early and use inplace=True
    

    #df['endbarrier_time'] = df['endbarrier_unix']
    #df['endbarrier_time'] = pd.to_datetime(df['endbarrier_unix'], unit='s')
    
    #df = df.drop(df.columns[-5], axis=1)
    

    #print('endbarrier',df.columns)

    df['endbarrier_unix'] = pd.to_datetime(df['endbarrier_unix'], unit='s')
    

  

    
   
        
    prices = df[['Close','touch_price', 'Date', 'endbarrier_unix', 'upper_barrier', 'lower_barrier']]
   
    

    df.drop(columns=[  'Date','unix','endbarrier_unix','Volume', 'Close', 'Volume', 'upper_barrier', 'lower_barrier', 'pct_change', 'Datetime', 'touch_price'], inplace=True)
    df.dropna(how='all', inplace=True)
    #df = df[lookback:]
    # Splitting data
    mapping = {-1: 0, 0: 1, 1: 2}

    df['label'] = df['label'].map(mapping)

    print('input dataframe:',df.columns)
    
    print('splitting data ...')
    train_datasets, test_datasets = crossvalidation.run_split_process(df)
   
    
    
    print('data dropped')
    
    feature_cols = df.drop('label',axis=1).columns

   # feature_cols = ['Close_AUDUSD', 'Close_USDCAD',
    #   'Close_USDCHF', 'AUDUSD_Returns', 'USDCAD_Returns', 'USDCHF_Returns']
    

    target_col = 'label'

    print('featurecols:',feature_cols)
    

    all_predictions = []
    all_actuals = []
    all_preds = []
    n_components = 15
    scaler = StandardScaler()
    
    # Define a parameter grid for GridSearchCV
    

    
    
    param_grid = {
        'n_estimators': [1000],
        'max_features': [4],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
       # 'bootstrap': [True, False]
    }

    # Training and Predicting for each split
   # for train_data, test_data, weight_data in zip(train_datasets[-1], test_datasets[-1], weights[-1]):
        #train = train_datasets
        #test = test_datasets
        #weight = weights[-1]

   
        
    train_idx =   train_datasets[-1]
   
   
    test_idx = test_datasets[-1]

    print('Test idx:', test_idx)

    train_data =df.iloc[train_idx]
    test_data = df.iloc[test_idx]

    startprice = prices['Close'].iloc[test_idx]

    endprice = prices['touch_price'].iloc[test_idx]
    Dates = prices['Date'].iloc[test_idx]
    enddate = prices['endbarrier_unix'].iloc[test_idx]
    upperbarrier = prices['upper_barrier'].iloc[test_idx]
    lowerbarrier = prices['lower_barrier'].iloc[test_idx]
    print('lowerbarrier', lowerbarrier)

    
  

   # weight_data =  weights[-1]
        
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]

    # Get the names of the columns with datetime dtype
   
#
    
    # Calculate class weights
   # 
    
    
    
    # Standardize the data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print('data scaled ...')
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)


    
    X_test = pca.transform(X_test)

    # Initialize GridSearchCV
    #clf = SVC(probability=True, C=50)
    clf =RandomForestClassifier(n_jobs =-1,random_state=44, n_estimators=1000,class_weight='balanced_subsample', criterion='entropy')
  #  clf = xgb.XGBClassifier(objective='multi:softmax',num_class=3, random_state=42)
    
    
    
   # clf = HistGradientBoostingClassifier()

    print('fitting model ...')
    clf.fit(X_train, y_train)
    
    print('model fitted ...')



    dum = DummyClassifier(strategy='stratified', random_state=0)

    dum.fit(X_train,y_train)
    dum_score =dum.score(X_test,y_test)

    print('Dumb Score:',dum_score)
    real_score =clf.score(X_test,y_test)
    print('Real Score:',real_score)


  


    ##########   Grid Search
    '''
    grid_search = GridSearchCV(clf, param_grid,refit=True,scoring='neg_log_loss', verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train, sample_weight=weight_data)

    best_log_loss = -grid_search.best_score_
    print(f"Best parameters found: {best_log_loss}")

    #grid_search.fit(X_train, y_train)

# Get the best estimator
    best_clf = grid_search.best_estimator_

# Use the best estimator to predict probabilities
    '''
    print('predicting ...')
    probas = clf.predict_proba(X_test)
    print('predicted ...')  
    

    
    #y_pred = (probas[:, 1] >= threshold).astype(int)

   # selected_columns= probas[:,[0,2]]
    max_proba_indices = np.argmax(probas, axis=1)
   # max_proba_indices= np.where(max_proba_indices==1,2,max_proba_indices)
    predicted_classes = clf.classes_[max_proba_indices]
    y_pred = predicted_classes

    


    # Print and store results
    print('######################')
    print('probas:', probas)
    print(classification_report(y_test, y_pred, zero_division=1))
    print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
    
    print(f'Y_true:{y_test} Y_pred:{y_pred}' )

    comparison_df = pd.DataFrame({'Y_true': y_test, 'Y_pred': y_pred})

    print(comparison_df)

   

    print(len(y_test))
    print(len(y_pred))
    print(len(probas[0]))
    print(len(probas[1]))
    print(len(probas[2]))
    print(probas.shape)


    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predictions': y_pred,
        'down proba': probas[:,0],
        'neutral proba': probas[:,1],
        'up proba': probas[:,2],
        'Dates':Dates  
    })

    predictions_df.to_csv('predictions_df.csv')
    all_predictions.append(predictions_df)

    all_actuals.extend(y_test.tolist())
    all_preds.extend(y_pred.tolist())
    print('###########################')
    #log_loss(y_test, y_pred)

    classes = np.unique(y_train)

# Compute log loss
    l_l = log_loss(y_test, probas, labels=classes)





    #l_l = log_loss(y_test, probas)
    print('logloss', l_l)  
  
    actual_ =[]
    prediction_ =[]
    dwn_ =[]
    neutral_ =[]
    up_ =[]
    start_ =[]
    end_ = []
    upper_ = []
    lower_ =[]
    date_ =[]
    enddate_=[]
    print(f"Length of actual_: {len(actual_)}")
    print(f"Length of prediction_: {len(prediction_)}")
    # Add similar print statements for dwn_, neutral_, up_, start_, end_, upper_, lower_, date_, enddate_

    # Ensure all lists have the same length before creating the DataFrame
    if not all(len(lst) == len(actual_) for lst in [prediction_, dwn_, neutral_, up_, start_, end_, upper_, lower_, date_, enddate_]):
        print("Not all lists have the same length. Check the loop logic and data sources.")
   
    '''
    else:
        predictions_df = pd.DataFrame({
            'Actual': actual_,
            'Predictions': prediction_,
            'down proba': dwn_,
            'neutral proba': neutral_,
            'up proba': up_,
            'start': start_,
            'end': end_,
            'upper': upper_,
            # Ensure all columns are included here
        })
    '''
 #   predictions_df.to_csv('predictions_df.csv')

    





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
    