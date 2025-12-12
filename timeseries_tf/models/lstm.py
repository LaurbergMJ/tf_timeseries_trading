from tensorflow import keras
from tensorflow.keras import layers

def build_lstm_forecaster(
        window_size: int, 
        horizon: int = 1, 
        lstm_units: int = 64,
        dense_units: int = 32,
        learning_rate: float = 1e-3,
    ) -> keras.Model:
    """
    Docstring for build_lstm_forecaster
    Simple LSTM model for forecasting 1D time series
    """

    inputs = keras.Input(shape=(window_size, 1))
    x = layers.LSTM(lstm_units)(inputs)
    x = layers.Dense(dense_units, activation='relu')(x)
    outputs = layers.Dense(horizon)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='lstm_forecaster')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )
    return model 