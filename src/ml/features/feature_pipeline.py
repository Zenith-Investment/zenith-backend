"""
Feature pipeline for data preprocessing and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import logging

from .technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Pipeline for feature engineering and data preprocessing.

    Handles:
    - Technical indicator calculation
    - Feature scaling/normalization
    - Missing value handling
    - Train/test splitting
    """

    def __init__(self, indicators_config: Optional[dict] = None):
        """
        Initialize feature pipeline.

        Args:
            indicators_config: Configuration for technical indicators
        """
        self.indicators_config = indicators_config or {}
        self.tech_indicators = TechnicalIndicators()
        self.feature_names: Optional[List[str]] = None
        self.scaler = None

    def prepare_features(
        self,
        df: pd.DataFrame,
        add_indicators: bool = True,
        handle_nan: bool = True
    ) -> pd.DataFrame:
        """
        Prepare features for ML models.

        Args:
            df: DataFrame with OHLCV data
            add_indicators: Whether to add technical indicators
            handle_nan: Whether to handle NaN values

        Returns:
            DataFrame with prepared features
        """
        df = df.copy()

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        # Add technical indicators
        if add_indicators:
            df = self.tech_indicators.add_all_indicators(df, self.indicators_config)

        # Add price-based features
        df = self._add_price_features(df)

        # Add volume-based features
        df = self._add_volume_features(df)

        # Add time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df = self._add_time_features(df)

        # Handle NaN values
        if handle_nan:
            df = self._handle_missing_values(df)

        self.feature_names = [col for col in df.columns if col not in required_cols]

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        df = df.copy()

        # Price changes
        df["price_change"] = df["close"].pct_change()
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        df["close_open_diff"] = (df["close"] - df["open"]) / df["open"]

        # Price momentum
        for period in [1, 5, 10, 20]:
            df[f"return_{period}d"] = df["close"].pct_change(period)

        # Volatility
        for period in [5, 10, 20]:
            df[f"volatility_{period}d"] = df["close"].pct_change().rolling(period).std()

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df = df.copy()

        # Volume changes
        df["volume_change"] = df["volume"].pct_change()

        # Volume moving averages
        for period in [5, 10, 20]:
            df[f"volume_ma_{period}"] = df["volume"].rolling(period).mean()
            df[f"volume_ratio_{period}"] = df["volume"] / df[f"volume_ma_{period}"]

        # Price-volume correlation
        df["price_volume_corr"] = df["close"].rolling(20).corr(df["volume"])

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()

        # Day of week (0=Monday, 6=Sunday)
        df["day_of_week"] = df.index.dayofweek

        # Month
        df["month"] = df.index.month

        # Quarter
        df["quarter"] = df.index.quarter

        # Is month end
        df["is_month_end"] = df.index.is_month_end.astype(int)

        # Is quarter end
        df["is_quarter_end"] = df.index.is_quarter_end.astype(int)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values and infinity values in the DataFrame.

        Strategy:
        - Replace infinity values with NaN
        - Clip extreme values to prevent float32 overflow
        - Forward fill for most features
        - Drop rows with remaining NaN in critical columns
        """
        df = df.copy()

        # Replace infinity values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Clip extreme values to prevent float32 overflow (max float32 is ~3.4e38)
        # Use a safe range: -1e10 to 1e10
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].clip(lower=-1e10, upper=1e10)

        # Forward fill
        df = df.fillna(method="ffill")

        # Backward fill for any remaining NaN at the start
        df = df.fillna(method="bfill")

        # Drop any remaining rows with NaN
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)

        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing values")

        return df

    def create_target_variable(
        self,
        df: pd.DataFrame,
        target_type: str = "binary_direction",
        prediction_horizon: int = 1,
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Create target variable for supervised learning.

        Args:
            df: DataFrame with price data
            target_type: Type of target variable
                - 'binary_direction': 1 if price goes up, 0 if down
                - 'multiclass': 0=down, 1=flat, 2=up
                - 'regression': actual price change
            prediction_horizon: Number of periods to look ahead
            threshold: Threshold for multiclass classification (% change)

        Returns:
            DataFrame with target variable added
        """
        df = df.copy()

        # Calculate future return
        df["future_return"] = df["close"].pct_change(prediction_horizon).shift(-prediction_horizon)

        if target_type == "binary_direction":
            df["target"] = (df["future_return"] > 0).astype(int)

        elif target_type == "multiclass":
            df["target"] = pd.cut(
                df["future_return"],
                bins=[-np.inf, -threshold, threshold, np.inf],
                labels=[0, 1, 2]
            ).astype(int)

        elif target_type == "regression":
            df["target"] = df["future_return"]

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        # Drop rows where we can't calculate future return
        df = df.dropna(subset=["target", "future_return"])

        return df

    def train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: DataFrame with features and target
            test_size: Proportion of data for test set
            validation_size: Proportion of data for validation set

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        test_start = int(n * (1 - test_size))
        val_start = int(n * (1 - test_size - validation_size))

        train_df = df.iloc[:val_start]
        val_df = df.iloc[val_start:test_start]
        test_df = df.iloc[test_start:]

        logger.info(
            f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        return train_df, val_df, test_df

    def get_feature_columns(self, exclude_cols: Optional[List[str]] = None) -> List[str]:
        """
        Get list of feature columns.

        Args:
            exclude_cols: Columns to exclude

        Returns:
            List of feature column names
        """
        if self.feature_names is None:
            raise ValueError("Features not prepared yet. Call prepare_features() first.")

        if exclude_cols is None:
            exclude_cols = ["target", "future_return"]

        return [col for col in self.feature_names if col not in exclude_cols]
