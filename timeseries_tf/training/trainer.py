from dataclasses import dataclass
from typing import Any, Dict, Optional 

from tensorflow import keras 
import tensorflow as tf

@dataclass 
class TrainingConfig:
    epochs: int = 50 
    batch_size: int = 64 
    validation_split: float = 0.2 
    patience: int = 10 # early stopping 

def train_model(
        model: keras.Model,
        X: Any,
        y: Any,
        config: TrainingConfig,
        use_tf_dataset: bool = False, 
    ) -> keras.callbacks.History:
    """
    Train a model on (X, y) 
    
    If use_tf_dataset is True, X and y should be tf.data.Datasets (train, val)    
    """

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.patience,
            restore_best_weights=True
        )
    ]

    if use_tf_dataset:
        train_ds, val_ds = X, y
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs, 
            callbacks=callbacks,
        )
    else:
        history = model.fit(
            X,
            y,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.validation_split,
            callbacks=callbacks,
        )
        
    return history