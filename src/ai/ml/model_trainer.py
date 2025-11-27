"""
Model Training Pipeline for financial ML models.

Handles:
- Data preparation and feature engineering
- Model training with cross-validation
- Model persistence and versioning
- Performance tracking
"""
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import structlog

from src.ai.ml.feature_engineering import FeatureEngineer, feature_engineer
from src.ai.ml.models import (
    EnsemblePredictor,
    LSTMPredictor,
    XGBoostPredictor,
    RandomForestPredictor,
    PredictionResult,
)
from src.core.config import settings

logger = structlog.get_logger()


@dataclass
class TrainingResult:
    """Result of model training."""
    model_name: str
    ticker: str
    training_date: datetime
    samples_used: int
    metrics: dict
    feature_importance: Optional[dict]
    status: str
    error: Optional[str] = None


class ModelTrainer:
    """
    Manages ML model training and predictions.

    Features:
    - Automatic feature engineering
    - Multi-model training
    - Model persistence
    - Performance tracking
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir or "/tmp/investai_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.feature_engineer = FeatureEngineer()
        self.trained_models: dict[str, EnsemblePredictor] = {}

    async def train_for_ticker(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        target_days: int = 5,
        force_retrain: bool = False,
    ) -> TrainingResult:
        """
        Train models for a specific ticker.

        Args:
            ticker: Asset ticker
            price_data: DataFrame with OHLCV data
            target_days: Days ahead to predict
            force_retrain: Force retraining even if model exists

        Returns:
            TrainingResult with training metrics
        """
        model_path = self.models_dir / f"{ticker}_ensemble.pkl"

        # Check if model exists and is recent
        if not force_retrain and model_path.exists():
            model_age = datetime.now().timestamp() - model_path.stat().st_mtime
            if model_age < 86400 * 7:  # Less than 7 days old
                logger.info(f"Loading existing model for {ticker}")
                self._load_model(ticker)
                return TrainingResult(
                    model_name="Ensemble",
                    ticker=ticker,
                    training_date=datetime.fromtimestamp(model_path.stat().st_mtime),
                    samples_used=0,
                    metrics={"status": "loaded_existing"},
                    feature_importance=None,
                    status="success",
                )

        # Prepare features
        logger.info(f"Training models for {ticker}")

        try:
            features = self.feature_engineer.create_features(price_data)
            target = self.feature_engineer.create_target(
                features,
                target_days=target_days,
                target_type="return",
            )

            # Remove NaN and align
            valid_idx = ~(features.isna().any(axis=1) | target.isna())
            features = features[valid_idx]
            target = target[valid_idx]

            if len(features) < 100:
                return TrainingResult(
                    model_name="Ensemble",
                    ticker=ticker,
                    training_date=datetime.now(),
                    samples_used=len(features),
                    metrics={},
                    feature_importance=None,
                    status="error",
                    error="Insufficient data for training (min 100 samples)",
                )

            # Get feature columns (exclude price columns)
            feature_cols = [
                col for col in features.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]

            X = features[feature_cols].values
            y = target.values

            # Train ensemble
            ensemble = EnsemblePredictor()
            training_metrics = ensemble.train(X, y, feature_names=feature_cols)

            # Save model
            self.trained_models[ticker] = ensemble
            self._save_model(ticker, ensemble, feature_cols)

            return TrainingResult(
                model_name=ensemble.name,
                ticker=ticker,
                training_date=datetime.now(),
                samples_used=len(X),
                metrics=training_metrics,
                feature_importance=ensemble.get_feature_importance(),
                status="success",
            )

        except Exception as e:
            logger.error(f"Training failed for {ticker}", error=str(e))
            return TrainingResult(
                model_name="Ensemble",
                ticker=ticker,
                training_date=datetime.now(),
                samples_used=0,
                metrics={},
                feature_importance=None,
                status="error",
                error=str(e),
            )

    async def predict(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        target_days: int = 5,
    ) -> PredictionResult:
        """
        Make price prediction for a ticker.

        Args:
            ticker: Asset ticker
            price_data: Recent OHLCV data
            target_days: Days ahead to predict

        Returns:
            PredictionResult with prediction details
        """
        # Ensure model is trained
        if ticker not in self.trained_models:
            # Try to load
            if not self._load_model(ticker):
                # Train new model
                result = await self.train_for_ticker(ticker, price_data, target_days)
                if result.status != "success":
                    raise ValueError(f"Failed to train model: {result.error}")

        ensemble = self.trained_models[ticker]
        feature_cols = self._load_feature_cols(ticker)

        # Create features
        features = self.feature_engineer.create_features(price_data)

        if feature_cols:
            # Ensure we have all required features
            missing = set(feature_cols) - set(features.columns)
            if missing:
                logger.warning(f"Missing features: {missing}")
                for col in missing:
                    features[col] = 0

            X = features[feature_cols].iloc[-1].values
        else:
            feature_cols = [
                col for col in features.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            X = features[feature_cols].iloc[-1].values

        current_price = float(price_data["close"].iloc[-1])

        return ensemble.predict_with_details(X, current_price)

    def _save_model(
        self,
        ticker: str,
        model: EnsemblePredictor,
        feature_cols: list,
    ):
        """Save trained model to disk."""
        model_path = self.models_dir / f"{ticker}_ensemble.pkl"
        cols_path = self.models_dir / f"{ticker}_features.json"

        try:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            with open(cols_path, "w") as f:
                json.dump(feature_cols, f)

            logger.info(f"Saved model for {ticker}")

        except Exception as e:
            logger.error(f"Failed to save model for {ticker}", error=str(e))

    def _load_model(self, ticker: str) -> bool:
        """Load trained model from disk."""
        model_path = self.models_dir / f"{ticker}_ensemble.pkl"

        if not model_path.exists():
            return False

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            self.trained_models[ticker] = model
            logger.info(f"Loaded model for {ticker}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model for {ticker}", error=str(e))
            return False

    def _load_feature_cols(self, ticker: str) -> Optional[list]:
        """Load feature columns for a ticker."""
        cols_path = self.models_dir / f"{ticker}_features.json"

        if not cols_path.exists():
            return None

        try:
            with open(cols_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    async def batch_train(
        self,
        tickers_data: dict[str, pd.DataFrame],
        target_days: int = 5,
    ) -> dict[str, TrainingResult]:
        """
        Train models for multiple tickers.

        Args:
            tickers_data: Dict mapping ticker -> price DataFrame
            target_days: Days ahead to predict

        Returns:
            Dict mapping ticker -> TrainingResult
        """
        results = {}

        for ticker, data in tickers_data.items():
            logger.info(f"Training model for {ticker}")
            result = await self.train_for_ticker(ticker, data, target_days)
            results[ticker] = result

        return results

    def get_model_info(self, ticker: str) -> Optional[dict]:
        """Get information about a trained model."""
        model_path = self.models_dir / f"{ticker}_ensemble.pkl"

        if not model_path.exists():
            return None

        return {
            "ticker": ticker,
            "model_file": str(model_path),
            "last_trained": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
            "is_loaded": ticker in self.trained_models,
        }

    def list_trained_models(self) -> list[dict]:
        """List all trained models."""
        models = []

        for model_file in self.models_dir.glob("*_ensemble.pkl"):
            ticker = model_file.stem.replace("_ensemble", "")
            info = self.get_model_info(ticker)
            if info:
                models.append(info)

        return models


class AutoMLTrainer:
    """
    Automated ML training with hyperparameter optimization.

    Uses advanced techniques:
    - Bayesian optimization for hyperparameters
    - Ensemble stacking
    - Feature selection
    """

    def __init__(self):
        self.best_params: dict[str, dict] = {}

    async def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "xgboost",
        n_trials: int = 50,
    ) -> dict:
        """
        Optimize hyperparameters using cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            model_type: Model type to optimize
            n_trials: Number of optimization trials

        Returns:
            Best hyperparameters
        """
        try:
            import optuna
            from sklearn.model_selection import cross_val_score

            def objective(trial):
                if model_type == "xgboost":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    }
                    model = XGBoostPredictor(**params)

                elif model_type == "random_forest":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "max_depth": trial.suggest_int("max_depth", 5, 20),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    }
                    model = RandomForestPredictor(**params)

                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                # Cross-validation score
                from sklearn.model_selection import TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=3)

                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    model.train(X_train, y_train)
                    preds = []
                    for i in range(len(X_val)):
                        pred, _ = model.predict(X_val[i])
                        preds.append(pred[0])

                    from sklearn.metrics import r2_score
                    scores.append(r2_score(y_val, preds))

                return np.mean(scores)

            # Run optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            self.best_params[model_type] = study.best_params
            return study.best_params

        except ImportError:
            logger.warning("Optuna not installed, using default parameters")
            return {}

    async def train_optimized(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        optimize: bool = True,
    ) -> TrainingResult:
        """
        Train with optimized hyperparameters.

        Args:
            ticker: Asset ticker
            price_data: OHLCV data
            optimize: Whether to run hyperparameter optimization

        Returns:
            TrainingResult
        """
        fe = FeatureEngineer()
        features = fe.create_features(price_data)
        target = fe.create_target(features, target_days=5, target_type="return")

        # Remove NaN
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_idx]
        target = target[valid_idx]

        feature_cols = [
            col for col in features.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]

        X = features[feature_cols].values
        y = target.values

        if optimize and len(X) > 200:
            # Optimize hyperparameters
            logger.info("Optimizing hyperparameters...")
            await self.optimize_hyperparameters(X, y, "xgboost")
            await self.optimize_hyperparameters(X, y, "random_forest")

        # Create ensemble with optimized models
        ensemble = EnsemblePredictor()

        # Override with optimized params if available
        if "xgboost" in self.best_params:
            ensemble.models["xgboost"] = XGBoostPredictor(**self.best_params["xgboost"])
        if "random_forest" in self.best_params:
            ensemble.models["random_forest"] = RandomForestPredictor(**self.best_params["random_forest"])

        # Train
        metrics = ensemble.train(X, y, feature_names=feature_cols)

        return TrainingResult(
            model_name=ensemble.name,
            ticker=ticker,
            training_date=datetime.now(),
            samples_used=len(X),
            metrics=metrics,
            feature_importance=ensemble.get_feature_importance(),
            status="success",
        )


# Global instances
model_trainer = ModelTrainer()
automl_trainer = AutoMLTrainer()
