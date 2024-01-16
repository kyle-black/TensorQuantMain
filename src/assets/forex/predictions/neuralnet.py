#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_1d_cnn_model(input_shape, num_classes):
    """
    Build a 1D CNN model for time-series data.

    Parameters:
    - input_shape (tuple): Shape of the input data (e.g., (100, 1) for 100 time steps with 1 feature per step).
    - num_classes (int): Number of output classes.

    Returns:
    - model: A TensorFlow Keras model.
    """

    model = Sequential()

    # First convolutional layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    # Second convolutional layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['presicion'])

    return model

# Example usage
# Assuming time-series data with 100 time steps and 1 feature per step, and 3 output classes
#model = build_1d_cnn_model(input_shape=(100, 1), num_classes=3)
#model.summary()  # This will print the summary of the model