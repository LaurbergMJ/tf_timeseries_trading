from tensorflow import keras 
from tensorflow.keras import layers 
from typing import Sequence

def build_mlp_forecaster(
        window_size: int,
        horizon: int = 1, 
        hidden_units: Sequence[int] = (64, 32),
        learning_rate: float = 1e-3
    ) -> keras.Model:
    """
    Simple mlp model for forecasting 1D time series 
    """ 

    inputs = keras.Input(shape=(window_size, 1))
    x = layers.Flatten()(inputs)

    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)

    outputs = layers.Dense(horizon)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='mlp_forecaster')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )
    return model 