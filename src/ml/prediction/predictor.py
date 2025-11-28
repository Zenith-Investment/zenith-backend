"""
Main predictor class that orchestrates ML predictions.

Combines data fetching, feature engineering, model inference, and caching.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta
import os

from ..data.data_fetcher import DataFetcher
from ..data.data_preprocessor import DataPreprocessor
from ..features.feature_pipeline import FeaturePipeline
from ..models.lstm_model import LSTMModel
from ..models.ensemble_model import EnsembleModel
from .cache import PredictionCache

logger = logging.getLogger(__name__)


class Predictor:
    """
    Main predictor class for stock price predictions.

    Handles the full prediction pipeline:
    1. Fetch historical data
    2. Engineer features
    3. Make predictions using loaded models
    4. Cache results
    """

    def __init__(
        self,
        models_path: str = "./models",
        redis_url: str = "redis://localhost:6379/0",
        cache_ttl: int = 3600,
        use_cache: bool = True
    ):
        """
        Initialize predictor.

        Args:
            models_path: Path to saved models
            redis_url: Redis connection URL
            cache_ttl: Cache TTL in seconds
            use_cache: Whether to use caching
        """
        self.models_path = models_path
        self.use_cache = use_cache

        # Initialize components
        self.data_fetcher = DataFetcher()
        self.data_preprocessor = DataPreprocessor()
        self.feature_pipeline = FeaturePipeline()

        # Initialize cache
        if use_cache:
            self.cache = PredictionCache(redis_url=redis_url, default_ttl=cache_ttl)
        else:
            self.cache = None

        # Model instances (loaded on demand)
        self.lstm_model: Optional[LSTMModel] = None
        self.ensemble_model: Optional[EnsembleModel] = None

    def load_models(
        self,
        lstm_path: Optional[str] = None,
        ensemble_path: Optional[str] = None
    ):
        """
        Load trained models.

        Args:
            lstm_path: Path to LSTM model (without extension)
            ensemble_path: Path to ensemble model (without extension)
        """
        if lstm_path:
            logger.info(f"Loading LSTM model from {lstm_path}")
            self.lstm_model = LSTMModel()
            self.lstm_model.load(lstm_path)

        if ensemble_path:
            logger.info(f"Loading ensemble model from {ensemble_path}")
            self.ensemble_model = EnsembleModel()
            self.ensemble_model.load(ensemble_path)

    def predict_price(
        self,
        ticker: str,
        horizon: int = 1,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Predict future price using LSTM model.

        Args:
            ticker: Stock ticker symbol
            horizon: Number of days to predict ahead
            use_cache: Whether to use cache

        Returns:
            Dictionary with prediction results
        """
        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(ticker, "lstm", horizon=horizon)
            if cached:
                return cached

        # Fetch and prepare data
        logger.info(f"Making LSTM prediction for {ticker}")

        try:
            # Fetch recent data
            df = self.data_fetcher.fetch_yfinance(ticker, period="1y", interval="1d")

            # Prepare features
            df = self.feature_pipeline.prepare_features(df)

            # Normalize data
            df_normalized, norm_params = self.data_preprocessor.normalize_data(
                df,
                method='minmax',
                columns=['close']
            )

            # Create sequence
            sequence_length = self.lstm_model.sequence_length if self.lstm_model else 60
            last_sequence = df_normalized[['close']].values[-sequence_length:]

            # Make prediction
            if self.lstm_model is None:
                raise ValueError("LSTM model not loaded")

            predictions = self.lstm_model.predict_next(last_sequence, n_steps=horizon)

            # Denormalize predictions
            predictions_denorm = predictions * (
                norm_params['close']['max'] - norm_params['close']['min']
            ) + norm_params['close']['min']

            # Get current price
            current_price = float(df['close'].iloc[-1])

            # Calculate predicted change
            predicted_changes = [
                {
                    "day": i + 1,
                    "price": float(pred),
                    "change_pct": float((pred - current_price) / current_price * 100)
                }
                for i, pred in enumerate(predictions_denorm)
            ]

            result = {
                "ticker": ticker,
                "model_type": "lstm",
                "current_price": current_price,
                "predictions": predicted_changes,
                "horizon": horizon,
                "timestamp": datetime.now().isoformat(),
                "confidence": "medium"  # You can calculate this based on model metrics
            }

            # Cache result
            if use_cache and self.cache:
                self.cache.set(ticker, "lstm", result, horizon=horizon)

            return result

        except Exception as e:
            logger.error(f"Error making LSTM prediction for {ticker}: {e}")
            raise

    def predict_signal(
        self,
        ticker: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Predict trading signal using ensemble model.

        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cache

        Returns:
            Dictionary with signal prediction
        """
        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(ticker, "ensemble")
            if cached:
                return cached

        logger.info(f"Making ensemble signal prediction for {ticker}")

        try:
            # Fetch data
            df = self.data_fetcher.fetch_yfinance(ticker, period="6mo", interval="1d")

            # Prepare features
            df = self.feature_pipeline.prepare_features(df)

            # Get latest features
            feature_cols = self.feature_pipeline.get_feature_columns()
            latest_features = df[feature_cols].iloc[-1:].values

            # Make prediction
            if self.ensemble_model is None:
                raise ValueError("Ensemble model not loaded")

            if self.ensemble_model.task == "classification":
                # Get signal (0=sell, 1=hold, 2=buy)
                signal = self.ensemble_model.predict(latest_features)[0]
                probabilities = self.ensemble_model.predict_proba(latest_features)[0]

                signal_map = {0: "sell", 1: "hold", 2: "buy"}

                result = {
                    "ticker": ticker,
                    "model_type": "ensemble",
                    "signal": signal_map.get(int(signal), "hold"),
                    "confidence": float(probabilities[int(signal)]),
                    "probabilities": {
                        "sell": float(probabilities[0]),
                        "hold": float(probabilities[1]) if len(probabilities) > 1 else 0.0,
                        "buy": float(probabilities[2]) if len(probabilities) > 2 else 0.0
                    },
                    "current_price": float(df['close'].iloc[-1]),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Regression - predict price change
                predicted_change = self.ensemble_model.predict(latest_features)[0]

                result = {
                    "ticker": ticker,
                    "model_type": "ensemble",
                    "predicted_change_pct": float(predicted_change),
                    "signal": "buy" if predicted_change > 1 else "sell" if predicted_change < -1 else "hold",
                    "current_price": float(df['close'].iloc[-1]),
                    "timestamp": datetime.now().isoformat()
                }

            # Cache result
            if use_cache and self.cache:
                self.cache.set(ticker, "ensemble", result)

            return result

        except Exception as e:
            logger.error(f"Error making ensemble prediction for {ticker}: {e}")
            raise

    def predict_comprehensive(
        self,
        ticker: str,
        horizon: int = 5,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make comprehensive prediction using all available models.

        Args:
            ticker: Stock ticker symbol
            horizon: Prediction horizon for price prediction
            use_cache: Whether to use cache

        Returns:
            Dictionary with all predictions
        """
        result = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat()
        }

        # Price prediction
        if self.lstm_model:
            try:
                price_pred = self.predict_price(ticker, horizon, use_cache)
                result["price_prediction"] = price_pred
            except Exception as e:
                logger.error(f"LSTM prediction failed: {e}")
                result["price_prediction"] = {"error": str(e)}

        # Signal prediction
        if self.ensemble_model:
            try:
                signal_pred = self.predict_signal(ticker, use_cache)
                result["signal_prediction"] = signal_pred
            except Exception as e:
                logger.error(f"Ensemble prediction failed: {e}")
                result["signal_prediction"] = {"error": str(e)}

        return result

    def invalidate_cache(self, ticker: Optional[str] = None):
        """
        Invalidate cache for a ticker or all cache.

        Args:
            ticker: Ticker to invalidate (None for all)
        """
        if not self.cache:
            return

        if ticker:
            self.cache.invalidate_ticker(ticker)
        else:
            self.cache.clear_all()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self.cache:
            return {"status": "disabled"}

        return self.cache.get_stats()
