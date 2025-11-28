"""
Ensemble model combining RandomForest and XGBoost.

Uses both models for robust signal classification and price prediction.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import json
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble model combining RandomForest and XGBoost.

    Can be used for both classification (signal prediction) and regression (price prediction).
    """

    def __init__(
        self,
        task: str = "classification",
        rf_params: Optional[Dict[str, Any]] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
        ensemble_method: str = "voting"
    ):
        """
        Initialize ensemble model.

        Args:
            task: Type of task ('classification' or 'regression')
            rf_params: Parameters for RandomForest
            xgb_params: Parameters for XGBoost
            ensemble_method: Method to combine predictions ('voting', 'averaging', 'weighted')
        """
        self.task = task
        self.ensemble_method = ensemble_method

        # Default parameters
        default_rf_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1
        }

        default_xgb_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1
        }

        self.rf_params = {**default_rf_params, **(rf_params or {})}
        self.xgb_params = {**default_xgb_params, **(xgb_params or {})}

        # Initialize models
        if task == "classification":
            self.rf_model = RandomForestClassifier(**self.rf_params)
            self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        elif task == "regression":
            self.rf_model = RandomForestRegressor(**self.rf_params)
            self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        else:
            raise ValueError(f"Unknown task: {task}. Use 'classification' or 'regression'")

        self.feature_names: Optional[List[str]] = None
        self.metadata = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train both RandomForest and XGBoost models.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Names of features

        Returns:
            Dictionary with training results
        """
        self.feature_names = feature_names

        logger.info(f"Training ensemble model ({self.task})...")
        logger.info(f"Training samples: {len(X_train)}")

        # Train RandomForest
        logger.info("Training RandomForest...")
        self.rf_model.fit(X_train, y_train)

        # Train XGBoost with early stopping if validation data provided
        logger.info("Training XGBoost...")
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.xgb_model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.xgb_model.fit(X_train, y_train)

        # Evaluate on training data
        train_metrics = self.evaluate(X_train, y_train, split_name="train")

        # Evaluate on validation data if provided
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val, split_name="validation")

        # Store metadata
        self.metadata = {
            "training_date": datetime.now().isoformat(),
            "n_samples_train": len(X_train),
            "n_samples_val": len(X_val) if X_val is not None else 0,
            "n_features": X_train.shape[1],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }

        logger.info("Training completed")

        return {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }

    def predict(self, X: np.ndarray, return_individual: bool = False) -> np.ndarray:
        """
        Make predictions using ensemble.

        Args:
            X: Input features
            return_individual: If True, return individual model predictions

        Returns:
            Ensemble predictions (or dict if return_individual=True)
        """
        # Get predictions from both models
        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)

        # Combine predictions
        if self.ensemble_method == "voting":
            # For classification: majority voting
            # For regression: averaging
            if self.task == "classification":
                ensemble_pred = np.round((rf_pred + xgb_pred) / 2).astype(int)
            else:
                ensemble_pred = (rf_pred + xgb_pred) / 2

        elif self.ensemble_method == "averaging":
            ensemble_pred = (rf_pred + xgb_pred) / 2

        elif self.ensemble_method == "weighted":
            # Weight by feature importance or validation performance
            # For simplicity, using equal weights here
            # In production, you'd use validation performance
            ensemble_pred = (rf_pred + xgb_pred) / 2

        else:
            ensemble_pred = (rf_pred + xgb_pred) / 2

        if return_individual:
            return {
                "ensemble": ensemble_pred,
                "random_forest": rf_pred,
                "xgboost": xgb_pred
            }

        return ensemble_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Args:
            X: Input features

        Returns:
            Probability predictions
        """
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification tasks")

        # Get probabilities from both models
        rf_proba = self.rf_model.predict_proba(X)
        xgb_proba = self.xgb_model.predict_proba(X)

        # Average probabilities
        ensemble_proba = (rf_proba + xgb_proba) / 2

        return ensemble_proba

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True targets
            split_name: Name of the split for logging

        Returns:
            Dictionary with metrics
        """
        predictions = self.predict(X)

        if self.task == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y, predictions)),
                "precision": float(precision_score(y, predictions, average='weighted', zero_division=0)),
                "recall": float(recall_score(y, predictions, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y, predictions, average='weighted', zero_division=0))
            }
        else:  # regression
            metrics = {
                "mse": float(mean_squared_error(y, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
                "mae": float(mean_absolute_error(y, predictions)),
                "r2": float(r2_score(y, predictions))
            }

        logger.info(f"{split_name.capitalize()} metrics: {metrics}")

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from both models.

        Returns:
            DataFrame with feature importances
        """
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.rf_model.n_features_in_)]
        else:
            feature_names = self.feature_names

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "rf_importance": self.rf_model.feature_importances_,
            "xgb_importance": self.xgb_model.feature_importances_
        })

        # Calculate average importance
        importance_df["avg_importance"] = (
            importance_df["rf_importance"] + importance_df["xgb_importance"]
        ) / 2

        # Sort by average importance
        importance_df = importance_df.sort_values("avg_importance", ascending=False)

        return importance_df

    def save(self, filepath: str):
        """
        Save models to file.

        Args:
            filepath: Path to save models (without extension)
        """
        # Save RandomForest
        joblib.dump(self.rf_model, f"{filepath}_rf.joblib")

        # Save XGBoost
        self.xgb_model.save_model(f"{filepath}_xgb.json")

        # Save metadata
        metadata_with_config = {
            **self.metadata,
            "task": self.task,
            "ensemble_method": self.ensemble_method,
            "rf_params": self.rf_params,
            "xgb_params": self.xgb_params,
            "feature_names": self.feature_names
        }

        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata_with_config, f, indent=2)

        logger.info(f"Ensemble model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load models from file.

        Args:
            filepath: Path to model files (without extension)
        """
        # Load RandomForest
        self.rf_model = joblib.load(f"{filepath}_rf.joblib")

        # Load XGBoost
        if self.task == "classification":
            self.xgb_model = xgb.XGBClassifier()
        else:
            self.xgb_model = xgb.XGBRegressor()

        self.xgb_model.load_model(f"{filepath}_xgb.json")

        # Load metadata
        try:
            with open(f"{filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)

            self.task = metadata.get("task", self.task)
            self.ensemble_method = metadata.get("ensemble_method", self.ensemble_method)
            self.feature_names = metadata.get("feature_names")
            self.metadata = metadata

        except FileNotFoundError:
            logger.warning(f"Metadata file not found at {filepath}_metadata.json")

        logger.info(f"Ensemble model loaded from {filepath}")
