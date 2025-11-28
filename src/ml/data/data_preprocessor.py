"""
Data preprocessor for cleaning and preparing stock data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess and clean stock price data."""

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values, duplicates, and outliers.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Remove duplicates
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} duplicate rows")

        # Sort by index (timestamp)
        df = df.sort_index()

        # Handle missing values
        df = DataPreprocessor._handle_missing_values(df)

        # Remove outliers
        df = DataPreprocessor._remove_outliers(df)

        return df

    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in OHLCV data."""
        df = df.copy()

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")

            # Forward fill (use previous values)
            df = df.fillna(method='ffill')

            # Backward fill for any remaining NaN at the start
            df = df.fillna(method='bfill')

            # Drop rows that still have NaN
            df = df.dropna()

        return df

    @staticmethod
    def _remove_outliers(
        df: pd.DataFrame,
        column: str = 'close',
        n_std: float = 5.0
    ) -> pd.DataFrame:
        """
        Remove outliers using standard deviation method.

        Args:
            df: DataFrame with price data
            column: Column to check for outliers
            n_std: Number of standard deviations for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()

        if column not in df.columns:
            return df

        # Calculate z-score
        mean = df[column].mean()
        std = df[column].std()

        # Mark outliers
        outliers = np.abs(df[column] - mean) > (n_std * std)

        if outliers.any():
            logger.warning(f"Found {outliers.sum()} outliers in {column}")
            df = df[~outliers]

        return df

    @staticmethod
    def resample_data(
        df: pd.DataFrame,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """
        Resample data to a different frequency.

        Args:
            df: DataFrame with OHLCV data
            freq: Frequency to resample to
                  'D'=daily, 'W'=weekly, 'M'=monthly, 'H'=hourly, etc.

        Returns:
            Resampled DataFrame
        """
        df = df.copy()

        # Resample OHLCV data
        resampled = pd.DataFrame()

        if 'open' in df.columns:
            resampled['open'] = df['open'].resample(freq).first()
        if 'high' in df.columns:
            resampled['high'] = df['high'].resample(freq).max()
        if 'low' in df.columns:
            resampled['low'] = df['low'].resample(freq).min()
        if 'close' in df.columns:
            resampled['close'] = df['close'].resample(freq).last()
        if 'volume' in df.columns:
            resampled['volume'] = df['volume'].resample(freq).sum()

        # Drop NaN rows
        resampled = resampled.dropna()

        logger.info(f"Resampled data from {len(df)} to {len(resampled)} rows")

        return resampled

    @staticmethod
    def normalize_data(
        df: pd.DataFrame,
        method: str = 'minmax',
        columns: Optional[list] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Normalize data for ML models.

        Args:
            df: DataFrame with features
            method: Normalization method ('minmax', 'standard', 'log')
            columns: Columns to normalize (default: all numeric columns)

        Returns:
            Tuple of (normalized DataFrame, normalization parameters)
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        params = {}

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
                params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}

            elif method == 'standard':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / (std + 1e-8)
                params[col] = {'method': 'standard', 'mean': mean, 'std': std}

            elif method == 'log':
                # Add 1 to handle zeros
                df[col] = np.log1p(df[col])
                params[col] = {'method': 'log'}

        return df, params

    @staticmethod
    def denormalize_data(
        df: pd.DataFrame,
        params: dict
    ) -> pd.DataFrame:
        """
        Denormalize data back to original scale.

        Args:
            df: Normalized DataFrame
            params: Normalization parameters from normalize_data()

        Returns:
            Denormalized DataFrame
        """
        df = df.copy()

        for col, param in params.items():
            if col not in df.columns:
                continue

            method = param['method']

            if method == 'minmax':
                df[col] = df[col] * (param['max'] - param['min']) + param['min']

            elif method == 'standard':
                df[col] = df[col] * param['std'] + param['mean']

            elif method == 'log':
                df[col] = np.expm1(df[col])

        return df

    @staticmethod
    def create_sequences(
        data: np.ndarray,
        sequence_length: int,
        target_column_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.

        Args:
            data: Array of features
            sequence_length: Length of input sequences
            target_column_index: Index of target column in data

        Returns:
            Tuple of (X sequences, y targets)
        """
        X, y = [], []

        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length, target_column_index])

        return np.array(X), np.array(y)

    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> dict:
        """
        Check data quality and return statistics.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with quality metrics
        """
        quality = {
            "total_rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max())
            },
            "columns": df.columns.tolist(),
        }

        # Check for zero values in volume
        if 'volume' in df.columns:
            zero_volume = (df['volume'] == 0).sum()
            quality["zero_volume_rows"] = int(zero_volume)

        # Check for negative values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                quality[f"negative_{col}"] = int(negative_count)

        return quality
