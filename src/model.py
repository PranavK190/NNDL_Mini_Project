import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

def build_mlp_model(input_dim=168):
    """
    Build a Multilayer Perceptron (MLP) model for hourly electricity forecasting.
    Input: Past 168 hourly observations (1 week of historical data).
    Output: Next 1 hour observation.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ], name="MLP_Electricity_Forecaster")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'rmse'] if hasattr(tf.keras.metrics, 'rmse') else ['mae']
    )
    return model

def build_lstm_model(input_dim=168):
    """
    Build a compact LSTM model for optional sequence comparison.
    Input shape: (168, 1)
    """
    model = Sequential([
        Input(shape=(input_dim, 1)),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ], name="LSTM_Electricity_Forecaster")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model
