import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def create_model(input_dim, num_classes):
    """
    Creates a 3-layer neural network with specified input dimension and number of output classes.

    Parameters:
    - input_dim: Integer, the size of the input layer (number of features).
    - num_classes: Integer, the number of classes for the output layer.

    Returns:
    - model: A Keras Sequential model.
    """
    model = Sequential([
        Dense(50, input_dim=input_dim, activation='relu'),  # First hidden layer
        Dense(100, activation='relu'),                      # Second hidden layer
        Dense(150, activation='relu'),                      # Third hidden layer
        Dense(num_classes, activation='softmax')            # Output layer
    ])
    return model

def compile_model(model):
    """
    Compiles the neural network model with an optimizer, loss function, and evaluation metrics.

    Parameters:
    - model: A Keras Sequential model.

    No return value.
    """
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

def train_model(model, X_train, y_train, epochs, batch_size):
    """
    Trains the model on the training data.

    Parameters:
    - model: A Keras Sequential model.
    - X_train: Feature data for training.
    - y_train: Labels for training.
    - epochs: Number of epochs to train the model.
    - batch_size: Batch size for training.

    Returns:
    - history: A history object containing training history metrics.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return history
