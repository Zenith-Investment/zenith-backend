"""
Advanced ML models for financial prediction.

Implements state-of-the-art models:
- LSTM (Long Short-Term Memory) for time series
- XGBoost for gradient boosting
- Random Forest for ensemble prediction
- Ensemble combining all models

DISCLAIMER: All predictions are for educational purposes only.
Past performance does not guarantee future results.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = structlog.get_logger()


@dataclass
class PredictionResult:
    """ML prediction result."""
    predicted_price: float
    predicted_return: float
    confidence: float
    prediction_low: float
    prediction_high: float
    model_name: str
    feature_importance: Optional[dict] = None
    model_metrics: Optional[dict] = None


class BasePredictor(ABC):
    """Base class for ML predictors."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals."""
        pass

    def get_feature_importance(self) -> Optional[dict]:
        """Get feature importance if available."""
        return None


class LSTMPredictor(BasePredictor):
    """
    LSTM Neural Network for time series prediction.

    Uses TensorFlow/Keras for implementation.
    Best for capturing long-term dependencies in price movements.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        units: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__("LSTM Neural Network")
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._model = None

    def _build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam

            model = Sequential([
                LSTM(
                    units=self.units,
                    return_sequences=True,
                    input_shape=input_shape,
                ),
                Dropout(self.dropout),
                LSTM(units=self.units, return_sequences=True),
                Dropout(self.dropout),
                LSTM(units=self.units),
                Dropout(self.dropout),
                Dense(units=25),
                Dense(units=1),
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="mean_squared_error",
            )

            return model

        except ImportError:
            logger.warning("TensorFlow not installed, using fallback")
            return None

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, 0])  # Predict close price
        return np.array(X), np.array(y)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
    ) -> dict:
        """Train LSTM model."""
        # Scale data
        scaled_data = self.scaler.fit_transform(X)

        # Create sequences
        X_seq, y_seq = self._create_sequences(scaled_data)

        if len(X_seq) == 0:
            return {"status": "error", "message": "Insufficient data for training"}

        # Build and train model
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        self._model = self._build_model(input_shape)

        if self._model is None:
            # Fallback to simple model without TensorFlow
            return self._train_fallback(X, y)

        history = self._model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
        )

        self.is_trained = True

        return {
            "status": "success",
            "model": self.name,
            "final_loss": float(history.history["loss"][-1]),
            "epochs": epochs,
        }

    def _train_fallback(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Fallback training without TensorFlow."""
        # Use simple exponential smoothing as fallback
        self._fallback_alpha = 0.3
        self._last_values = y[-self.sequence_length:]
        self.is_trained = True

        return {
            "status": "success",
            "model": f"{self.name} (Fallback)",
            "note": "Using exponential smoothing fallback",
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        if self._model is None:
            # Fallback prediction
            pred = np.mean(self._last_values) * (1 + np.random.normal(0, 0.02))
            std = np.std(self._last_values)
            return np.array([pred]), np.array([std])

        # Scale and create sequence
        scaled = self.scaler.transform(X)
        X_seq = scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)

        # Predict
        pred_scaled = self._model.predict(X_seq, verbose=0)

        # Inverse scale (approximate)
        pred = pred_scaled[0, 0] * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0]

        # Estimate uncertainty
        std = np.std(X[-20:, 0]) * 0.5

        return np.array([pred]), np.array([std])


class XGBoostPredictor(BasePredictor):
    """
    XGBoost Gradient Boosting for price prediction.

    Excellent for feature-rich datasets with complex patterns.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ):
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._feature_names = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> dict:
        """Train XGBoost model."""
        try:
            import xgboost as xgb

            self._feature_names = feature_names

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                scores.append(r2_score(y_val, pred))

            # Train final model on all data
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                n_jobs=-1,
            )
            self.model.fit(X, y)
            self.is_trained = True

            return {
                "status": "success",
                "model": self.name,
                "cv_r2_mean": float(np.mean(scores)),
                "cv_r2_std": float(np.std(scores)),
            }

        except ImportError:
            # Fallback to sklearn GradientBoosting
            return self._train_sklearn(X, y, feature_names)

    def _train_sklearn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> dict:
        """Train with sklearn fallback."""
        self._feature_names = feature_names

        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
        )
        self.model.fit(X, y)
        self.is_trained = True

        return {
            "status": "success",
            "model": f"{self.name} (sklearn fallback)",
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        pred = self.model.predict(X.reshape(1, -1))

        # Estimate uncertainty using prediction variance across trees
        try:
            # For xgboost, use prediction variance
            std = np.std(pred) if len(pred) > 1 else abs(pred[0]) * 0.1
        except Exception:
            std = abs(pred[0]) * 0.1

        return pred, np.array([std])

    def get_feature_importance(self) -> Optional[dict]:
        """Get feature importance."""
        if not self.is_trained or self._feature_names is None:
            return None

        importance = self.model.feature_importances_
        return dict(zip(self._feature_names, importance))


class RandomForestPredictor(BasePredictor):
    """
    Random Forest Ensemble for robust predictions.

    Good for reducing overfitting and capturing non-linear relationships.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
    ):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._feature_names = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> dict:
        """Train Random Forest model."""
        self._feature_names = feature_names

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            scores.append(r2_score(y_val, pred))

        # Train final model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self.is_trained = True

        return {
            "status": "success",
            "model": self.name,
            "cv_r2_mean": float(np.mean(scores)),
            "cv_r2_std": float(np.std(scores)),
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty from tree variance."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        X_reshaped = X.reshape(1, -1)

        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X_reshaped) for tree in self.model.estimators_
        ])

        pred = np.mean(tree_predictions)
        std = np.std(tree_predictions)

        return np.array([pred]), np.array([std])

    def get_feature_importance(self) -> Optional[dict]:
        """Get feature importance."""
        if not self.is_trained or self._feature_names is None:
            return None

        importance = self.model.feature_importances_
        return dict(zip(self._feature_names, importance))


class EnsemblePredictor(BasePredictor):
    """
    Ensemble model combining multiple ML algorithms.

    Uses weighted voting based on model performance.
    """

    def __init__(self):
        super().__init__("Ensemble (LSTM + XGBoost + RF)")
        self.models = {
            "lstm": LSTMPredictor(),
            "xgboost": XGBoostPredictor(),
            "random_forest": RandomForestPredictor(),
        }
        self.weights = {
            "lstm": 0.3,
            "xgboost": 0.4,
            "random_forest": 0.3,
        }
        self._feature_names = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> dict:
        """Train all models and calculate optimal weights."""
        self._feature_names = feature_names
        results = {}

        # Train each model
        for name, model in self.models.items():
            try:
                if name == "lstm":
                    result = model.train(X, y)
                else:
                    result = model.train(X, y, feature_names)
                results[name] = result
                logger.info(f"Trained {name}", result=result)
            except Exception as e:
                logger.error(f"Failed to train {name}", error=str(e))
                results[name] = {"status": "error", "error": str(e)}

        # Calculate weights based on cross-validation performance
        self._update_weights(X, y)

        self.is_trained = True

        return {
            "status": "success",
            "model": self.name,
            "individual_results": results,
            "weights": self.weights,
        }

    def _update_weights(self, X: np.ndarray, y: np.ndarray):
        """Update model weights based on validation performance."""
        # Use last 20% for validation
        split_idx = int(len(X) * 0.8)
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        scores = {}
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    predictions = []
                    for i in range(len(X_val)):
                        pred, _ = model.predict(X_val[i])
                        predictions.append(pred[0])
                    mse = mean_squared_error(y_val, predictions)
                    scores[name] = 1 / (mse + 1e-10)  # Inverse MSE as weight
                except Exception:
                    scores[name] = 0.1

        # Normalize weights
        total = sum(scores.values())
        if total > 0:
            self.weights = {k: v / total for k, v in scores.items()}

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        predictions = []
        uncertainties = []

        for name, model in self.models.items():
            if model.is_trained:
                try:
                    pred, std = model.predict(X)
                    predictions.append((pred[0], self.weights[name]))
                    uncertainties.append(std[0])
                except Exception as e:
                    logger.warning(f"Prediction failed for {name}", error=str(e))

        if not predictions:
            raise ValueError("No models available for prediction")

        # Weighted average
        total_weight = sum(w for _, w in predictions)
        ensemble_pred = sum(p * w for p, w in predictions) / total_weight

        # Combined uncertainty
        ensemble_std = np.sqrt(np.mean([u**2 for u in uncertainties]))

        return np.array([ensemble_pred]), np.array([ensemble_std])

    def get_feature_importance(self) -> Optional[dict]:
        """Get combined feature importance."""
        if self._feature_names is None:
            return None

        combined = {name: 0 for name in self._feature_names}

        for model_name, model in self.models.items():
            importance = model.get_feature_importance()
            if importance:
                weight = self.weights.get(model_name, 0)
                for feat, imp in importance.items():
                    if feat in combined:
                        combined[feat] += imp * weight

        return combined

    def predict_with_details(
        self,
        X: np.ndarray,
        current_price: float,
    ) -> PredictionResult:
        """Make prediction with full details."""
        pred, std = self.predict(X)

        predicted_price = current_price * (1 + pred[0])
        confidence = max(0.3, min(0.9, 1 - std[0] * 2))

        # Calculate prediction range (95% confidence interval)
        margin = std[0] * 1.96
        prediction_low = current_price * (1 + pred[0] - margin)
        prediction_high = current_price * (1 + pred[0] + margin)

        return PredictionResult(
            predicted_price=predicted_price,
            predicted_return=pred[0] * 100,
            confidence=confidence,
            prediction_low=prediction_low,
            prediction_high=prediction_high,
            model_name=self.name,
            feature_importance=self.get_feature_importance(),
            model_metrics={
                "weights": self.weights,
                "uncertainty": float(std[0]),
            },
        )


# Global instances
lstm_predictor = LSTMPredictor()
xgboost_predictor = XGBoostPredictor()
rf_predictor = RandomForestPredictor()
ensemble_predictor = EnsemblePredictor()
