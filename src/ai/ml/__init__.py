"""
Machine Learning module for advanced price prediction and analysis.

Implements state-of-the-art ML models:
- LSTM Neural Networks for time series
- XGBoost for feature-based prediction
- Random Forest ensemble
- Prophet for trend forecasting

DISCLAIMER: All predictions are for educational purposes only.
Past performance does not guarantee future results.
"""
from src.ai.ml.models import (
    LSTMPredictor,
    XGBoostPredictor,
    RandomForestPredictor,
    EnsemblePredictor,
)
from src.ai.ml.feature_engineering import FeatureEngineer
from src.ai.ml.model_trainer import ModelTrainer

__all__ = [
    "LSTMPredictor",
    "XGBoostPredictor",
    "RandomForestPredictor",
    "EnsemblePredictor",
    "FeatureEngineer",
    "ModelTrainer",
]
