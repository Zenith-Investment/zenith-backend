"""
LSTM model for time series forecasting.

Uses TensorFlow/Keras to build and train LSTM neural networks
for predicting stock price movements.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import logging
from datetime import datetime
import json

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
except ImportError:
    raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

logger = logging.getLogger(__name__)


class LSTMModel:
    """LSTM model for stock price prediction."""

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 1,
        lstm_units: List[int] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model.

        Args:
            sequence_length: Number of time steps to look back
            n_features: Number of input features
            lstm_units: List of LSTM units for each layer (default: [50, 50])
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units or [50, 50]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model: Optional[keras.Model] = None
        self.history = None
        self.metadata = {}

    def build_model(self) -> keras.Model:
        """
        Build LSTM model architecture.

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential(name="LSTM_Stock_Predictor")

        # Input layer
        model.add(layers.Input(shape=(self.sequence_length, self.n_features)))

        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            # Return sequences for all layers except the last
            return_sequences = i < len(self.lstm_units) - 1

            model.add(layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                name=f"lstm_{i+1}"
            ))

            # Add dropout for regularization
            model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{i+1}"))

        # Dense layers for output
        model.add(layers.Dense(25, activation='relu', name='dense_1'))
        model.add(layers.Dropout(self.dropout_rate, name='dropout_output'))
        model.add(layers.Dense(1, name='output'))

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model = model
        logger.info(f"Built LSTM model with {model.count_params()} parameters")

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: int = 1
    ) -> dict:
        """
        Train the LSTM model.

        Args:
            X_train: Training sequences (samples, sequence_length, n_features)
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping
            verbose: Verbosity level

        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=verbose
            )
        ]

        # Train model
        logger.info(f"Training LSTM model for up to {epochs} epochs...")

        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        self.history = history.history

        # Store metadata
        self.metadata = {
            "training_date": datetime.now().isoformat(),
            "n_samples_train": len(X_train),
            "n_samples_val": len(X_val) if X_val is not None else 0,
            "epochs_trained": len(history.history['loss']),
            "final_train_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
        }

        logger.info(f"Training completed in {self.metadata['epochs_trained']} epochs")
        logger.info(f"Final train loss: {self.metadata['final_train_loss']:.6f}")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input sequences (samples, sequence_length, n_features)

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        predictions = self.model.predict(X, verbose=0)
        return predictions

    def predict_next(self, last_sequence: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """
        Predict next n steps iteratively.

        Args:
            last_sequence: Last sequence of data (sequence_length, n_features)
            n_steps: Number of steps to predict ahead

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(n_steps):
            # Reshape for prediction
            input_seq = current_sequence.reshape(1, self.sequence_length, self.n_features)

            # Predict next value
            next_pred = self.model.predict(input_seq, verbose=0)[0, 0]
            predictions.append(next_pred)

            # Update sequence (shift and append prediction)
            # For multivariate, we only update the first feature (close price)
            new_row = current_sequence[-1].copy()
            new_row[0] = next_pred  # Assuming first feature is the target

            current_sequence = np.vstack([current_sequence[1:], new_row])

        return np.array(predictions)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance.

        Args:
            X_test: Test sequences
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        # Get predictions
        predictions = self.predict(X_test)

        # Calculate metrics
        mse = np.mean((predictions.flatten() - y_test) ** 2)
        mae = np.mean(np.abs(predictions.flatten() - y_test))
        rmse = np.sqrt(mse)

        # Direction accuracy (for price movements)
        y_diff = np.diff(y_test)
        pred_diff = np.diff(predictions.flatten())
        direction_accuracy = np.mean(np.sign(y_diff) == np.sign(pred_diff))

        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "direction_accuracy": float(direction_accuracy),
            "n_samples": len(y_test)
        }

        logger.info(f"Evaluation metrics: {metrics}")

        return metrics

    def save(self, filepath: str):
        """
        Save model to file.

        Args:
            filepath: Path to save model (without extension)
        """
        if self.model is None:
            raise ValueError("No model to save")

        # Save model architecture and weights
        self.model.save(f"{filepath}.keras")

        # Save metadata
        metadata_with_config = {
            **self.metadata,
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
        }

        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata_with_config, f, indent=2)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load model from file.

        Args:
            filepath: Path to model file (without extension)
        """
        # Load model
        self.model = keras.models.load_model(f"{filepath}.keras")

        # Load metadata
        try:
            with open(f"{filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)

            self.sequence_length = metadata.get("sequence_length", self.sequence_length)
            self.n_features = metadata.get("n_features", self.n_features)
            self.lstm_units = metadata.get("lstm_units", self.lstm_units)
            self.dropout_rate = metadata.get("dropout_rate", self.dropout_rate)
            self.learning_rate = metadata.get("learning_rate", self.learning_rate)
            self.metadata = metadata

        except FileNotFoundError:
            logger.warning(f"Metadata file not found at {filepath}_metadata.json")

        logger.info(f"Model loaded from {filepath}")

    def get_model_summary(self) -> str:
        """
        Get model architecture summary.

        Returns:
            String representation of model
        """
        if self.model is None:
            return "Model not built"

        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
