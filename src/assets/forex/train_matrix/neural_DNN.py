import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
import crossvalidation
import joblib
import random

# Ensure seed is set for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# Custom wrapper to use neural network model with CalibratedClassifierCV
class ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.classes_ = None  # This will be set during fit

    def fit(self, X, y):
        # Set the classes_ attribute from the unique labels in the target
        self.classes_ = np.unique(y)
        # No need to fit the model again since it's already trained
        return self

    def predict_proba(self, X):
        # Ensure that this method returns the predicted probabilities
        return self.model.predict(X)

# Model training function with isotonic regression
def run_model(df, asset, lookback, n_components, training_cols, model_num, learning_rate=0.001, batch_size=128, epochs=300, seed=42):
    set_seed(seed)
    startlookback = lookback * 10

    df.dropna(inplace=True)
    df.dropna(how='all', inplace=True)
    df = df[startlookback:]
    df['endbarrier_unix'] = pd.to_datetime(df['endbarrier_unix'], unit='s')
    prices = df[['Close', 'touch_price', 'Date', 'endbarrier_unix', 'upper_barrier', 'lower_barrier', 'pct', 'touch_time_unix']]
    df = df[training_cols]

    # Ensure cross-validation is consistent with the set seed
    train_datasets, test_datasets = crossvalidation.run_split_process(df)
    feature_cols = df.drop('label', axis=1).columns
    target_col = 'label'

    n_components = 17
    scaler = StandardScaler()

    train_idx = train_datasets[-1]
    test_idx = test_datasets[-1]

    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Standardize the data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert y_train and y_test to class labels (0 and 1)
    y_train_labels = y_train.values  # Assuming y_train is a pandas Series
    y_test_labels = y_test.values

    # One-hot encode for model training
    y_train_onehot = tf.keras.utils.to_categorical(y_train_labels, num_classes=2)
    y_test_onehot = tf.keras.utils.to_categorical(y_test_labels, num_classes=2)

    # Build and train the neural network model
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

    model.add(layers.Dense(2, activation='softmax'))  # Use softmax for multi-class probabilities

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',  # Use categorical_crossentropy with softmax
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=1e-6)

    model.fit(X_train, y_train_onehot, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, lr_scheduler])

    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test_onehot)
    print(f'Test accuracy: {test_acc}')
    print(f'Test precision: {test_precision}')
    print(f'Test recall: {test_recall}')
    print(f'Test AUC: {test_auc}')

    # Get predicted probabilities
    y_pred_proba = model.predict(X_test)

    # Wrap the trained model
    wrapped_model = ModelWrapper(model)

    # **Important**: Fit the wrapped model to set `classes_`
    wrapped_model.fit(X_train, y_train_labels)

    # Ensure `classes_` is set
    print("Classes in wrapped_model:", wrapped_model.classes_)

    # Apply Isotonic Regression for calibration
    calibrated_model = CalibratedClassifierCV(estimator=wrapped_model, method='isotonic', cv='prefit')

    # Fit the calibration model using the training set
    calibrated_model.fit(X_train, y_train_labels)

    # Get the calibrated probabilities
    y_pred_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

    # Calculate log loss for the calibrated model
    log_loss_value_calibrated = log_loss(y_test_labels, y_pred_proba_calibrated)
    print(f'Log Loss (Calibrated): {log_loss_value_calibrated}')

    # Calculate Brier score for the calibrated model
    brier_score_calibrated = brier_score_loss(y_test_labels, y_pred_proba_calibrated)
    print(f'Brier Score (Calibrated): {brier_score_calibrated}')

    # Save the scaler, model, and PCA
    joblib.dump(scaler, f'../deploy/models/EURUSD/{model_num}_scaler.pkl')
    model.save(f'../deploy/models/EURUSD/{model_num}.h5')
    joblib.dump(pca, f'../deploy/models/EURUSD/{model_num}_pca.pkl')

    return model, y_pred_proba_calibrated, y_test_labels