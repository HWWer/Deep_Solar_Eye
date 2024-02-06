import optuna
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def split_data(df, images, test_size=0.2, random_state=42):
    X_train, X_valid, y_train, y_valid = train_test_split(images, df['I'].values, test_size=test_size, random_state=random_state)
    return X_train, X_valid, y_train, y_valid

def initialize_model(input_shape, ne):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))

    return model

def compile_model(model, learning_rate=0.001):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

def objective(trial, x_train, y_train, patience, epochs):
    input_shape = x_train[0].shape

    model = initialize_model(input_shape, trial.suggest_int('ne', 32, 128))

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
    batch_size = trial.suggest_int('batch_size', 8, 32)

    model = compile_model(model, learning_rate=learning_rate)

    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    return -history.history['val_loss'][-1]

def optimize(trials, x_train_norm, y_train, patience, epochs):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, x_train_norm, y_train, patience, epochs), n_trials=trials)

    return study

