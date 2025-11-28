"""
Model trainer for ML models.

Handles the complete training pipeline for LSTM and ensemble models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging
from datetime import datetime
import os

from ..data.data_fetcher import DataFetcher
from ..data.data_preprocessor import DataPreprocessor
from ..features.feature_pipeline import FeaturePipeline
from ..models.lstm_model import LSTMModel
from ..models.ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trainer for ML models."""

    def __init__(
        self,
        models_path: str = "./models",
        data_period: str = "1y"
    ):
        """
        Initialize model trainer.

        Args:
            models_path: Path to save trained models
            data_period: Period of historical data to use
        """
        self.models_path = models_path
        self.data_period = data_period

        # Create models directory if it doesn't exist
        os.makedirs(models_path, exist_ok=True)

        # Initialize components
        self.data_fetcher = DataFetcher()
        self.data_preprocessor = DataPreprocessor()
        self.feature_pipeline = FeaturePipeline()

    def prepare_data_for_lstm(
        self,
        ticker: str,
        sequence_length: int = 60,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Prepare data for LSTM training.

        Args:
            ticker: Stock ticker symbol
            sequence_length: Length of input sequences
            test_size: Proportion for test set
            val_size: Proportion for validation set

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, metadata)
        """
        logger.info(f"Preparing LSTM data for {ticker}")

        # Fetch data
        df = self.data_fetcher.fetch_yfinance(ticker, period=self.data_period)

        # Clean data
        df = self.data_preprocessor.clean_data(df)

        # For LSTM, we typically just use the close price
        # You can extend this to use more features
        close_prices = df[['close']].values

        # Normalize
        df_norm, norm_params = self.data_preprocessor.normalize_data(
            pd.DataFrame(close_prices, columns=['close']),
            method='minmax'
        )

        normalized_data = df_norm.values

        # Create sequences
        X, y = self.data_preprocessor.create_sequences(
            normalized_data,
            sequence_length=sequence_length,
            target_column_index=0
        )

        # Split data
        n = len(X)
        test_start = int(n * (1 - test_size))
        val_start = int(n * (1 - test_size - val_size))

        X_train = X[:val_start]
        y_train = y[:val_start]
        X_val = X[val_start:test_start]
        y_val = y[val_start:test_start]
        X_test = X[test_start:]
        y_test = y[test_start:]

        metadata = {
            "ticker": ticker,
            "sequence_length": sequence_length,
            "n_samples": n,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "normalization_params": norm_params,
            "data_period": self.data_period
        }

        logger.info(f"Data prepared: {n} samples, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        return X_train, y_train, X_val, y_val, X_test, y_test, metadata

    def train_lstm_model(
        self,
        ticker: str,
        sequence_length: int = 60,
        lstm_units: Optional[list] = None,
        epochs: int = 100,
        batch_size: int = 32,
        save_model: bool = True
    ) -> Tuple[LSTMModel, dict]:
        """
        Train LSTM model.

        Args:
            ticker: Stock ticker symbol
            sequence_length: Length of input sequences
            lstm_units: List of LSTM units per layer
            epochs: Number of training epochs
            batch_size: Batch size
            save_model: Whether to save the trained model

        Returns:
            Tuple of (trained model, evaluation metrics)
        """
        logger.info(f"Training LSTM model for {ticker}")

        # Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test, metadata = \
            self.prepare_data_for_lstm(ticker, sequence_length)

        # Initialize model
        model = LSTMModel(
            sequence_length=sequence_length,
            n_features=X_train.shape[2],
            lstm_units=lstm_units or [50, 50]
        )

        # Train
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )

        # Evaluate
        metrics = model.evaluate(X_test, y_test)

        # Save model
        model_path = None
        if save_model:
            model_path = os.path.join(self.models_path, f"lstm_{ticker}_{datetime.now().strftime('%Y%m%d')}")
            model.save(model_path)
            # Note: model.save() adds .keras extension, but we return path without it
            # because model.load() expects path without extension
            logger.info(f"Model saved to {model_path}.keras")

        return model, metrics, model_path

    def prepare_data_for_ensemble(
        self,
        ticker: str,
        target_type: str = "binary_direction",
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Prepare data for ensemble training.

        Args:
            ticker: Stock ticker symbol
            target_type: Type of target variable
            test_size: Proportion for test set
            val_size: Proportion for validation set

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
        """
        logger.info(f"Preparing ensemble data for {ticker}")

        # Fetch data
        df = self.data_fetcher.fetch_yfinance(ticker, period=self.data_period)

        # Prepare features
        df = self.feature_pipeline.prepare_features(df)

        # Create target variable
        df = self.feature_pipeline.create_target_variable(
            df,
            target_type=target_type,
            prediction_horizon=1
        )

        # Split data
        train_df, val_df, test_df = self.feature_pipeline.train_test_split(
            df,
            test_size=test_size,
            validation_size=val_size
        )

        # Get features
        feature_cols = self.feature_pipeline.get_feature_columns()

        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['target'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values

        logger.info(f"Data prepared: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

    def train_ensemble_model(
        self,
        ticker: str,
        task: str = "classification",
        target_type: str = "binary_direction",
        save_model: bool = True
    ) -> Tuple[EnsembleModel, dict]:
        """
        Train ensemble model.

        Args:
            ticker: Stock ticker symbol
            task: Task type ('classification' or 'regression')
            target_type: Type of target variable
            save_model: Whether to save the trained model

        Returns:
            Tuple of (trained model, evaluation metrics)
        """
        logger.info(f"Training ensemble model for {ticker}")

        # Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = \
            self.prepare_data_for_ensemble(ticker, target_type)

        # Initialize model
        model = EnsembleModel(task=task)

        # Train
        training_results = model.train(
            X_train, y_train,
            X_val, y_val,
            feature_names=feature_names
        )

        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test, split_name="test")

        # Get feature importance
        feature_importance = model.get_feature_importance()
        logger.info(f"Top 5 important features:\n{feature_importance.head()}")

        # Save model
        model_path = None
        if save_model:
            model_path = os.path.join(self.models_path, f"ensemble_{ticker}_{datetime.now().strftime('%Y%m%d')}")
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")

        return model, test_metrics, model_path

    def train_all_models(
        self,
        ticker: str,
        lstm_params: Optional[Dict[str, Any]] = None,
        ensemble_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train all models for a ticker.

        Args:
            ticker: Stock ticker symbol
            lstm_params: Parameters for LSTM training
            ensemble_params: Parameters for ensemble training

        Returns:
            Dictionary with training results
        """
        results = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat()
        }

        # Train LSTM
        try:
            logger.info("Training LSTM model...")
            lstm_model, lstm_metrics = self.train_lstm_model(
                ticker,
                **(lstm_params or {})
            )
            results["lstm"] = {
                "status": "success",
                "metrics": lstm_metrics
            }
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            results["lstm"] = {
                "status": "failed",
                "error": str(e)
            }

        # Train ensemble
        try:
            logger.info("Training ensemble model...")
            ensemble_model, ensemble_metrics = self.train_ensemble_model(
                ticker,
                **(ensemble_params or {})
            )
            results["ensemble"] = {
                "status": "success",
                "metrics": ensemble_metrics
            }
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            results["ensemble"] = {
                "status": "failed",
                "error": str(e)
            }

        return results
