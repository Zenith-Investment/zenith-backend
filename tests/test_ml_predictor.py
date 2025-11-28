"""
Unit tests for ML Predictor.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.ml.prediction.predictor import Predictor


@pytest.fixture
def mock_data_fetcher():
    """Mock DataFetcher."""
    mock = Mock()
    mock_df = Mock()
    mock_df.shape = (100, 6)
    mock_df.columns = ['open', 'high', 'low', 'close', 'volume', 'date']
    mock.fetch_yfinance.return_value = mock_df
    return mock


@pytest.fixture
def predictor(mock_data_fetcher):
    """Create predictor instance with mocked dependencies."""
    with patch('src.ml.prediction.predictor.DataFetcher', return_value=mock_data_fetcher):
        with patch('src.ml.prediction.predictor.PredictionCache'):
            return Predictor(use_cache=False)


def test_predictor_initialization(predictor):
    """Test predictor initialization."""
    assert predictor is not None
    assert predictor.feature_pipeline is not None


def test_predictor_load_models(predictor):
    """Test loading ML models."""
    with patch('src.ml.prediction.predictor.LSTMModel') as mock_lstm:
        with patch('src.ml.prediction.predictor.EnsembleModel') as mock_ensemble:
            predictor.load_models(lstm_path="/fake/lstm", ensemble_path="/fake/ensemble")

            assert predictor.lstm_model is not None
            assert predictor.ensemble_model is not None


def test_predict_price_without_model():
    """Test price prediction fails without loaded model."""
    predictor = Predictor(use_cache=False)

    with pytest.raises(ValueError, match="LSTM model not loaded"):
        predictor.predict_price("PETR4.SA", horizon=5)


def test_predict_signal_without_model():
    """Test signal prediction fails without loaded model."""
    predictor = Predictor(use_cache=False)

    with pytest.raises(ValueError, match="Ensemble model not loaded"):
        predictor.predict_signal("PETR4.SA")


@patch('src.ml.prediction.predictor.DataFetcher')
def test_predict_price_with_mock_model(mock_fetcher_class, mock_data_fetcher):
    """Test price prediction with mocked model."""
    mock_fetcher_class.return_value = mock_data_fetcher

    predictor = Predictor(use_cache=False)

    # Mock LSTM model
    mock_lstm = Mock()
    mock_lstm.predict_next.return_value = np.array([[30.5], [31.2], [31.8], [32.1], [32.5]])
    predictor.lstm_model = mock_lstm

    # Mock feature pipeline
    mock_df = Mock()
    mock_df.iloc.__getitem__.return_value = {'close': 30.0}
    predictor.feature_pipeline.prepare_features.return_value = mock_df

    result = predictor.predict_price("PETR4.SA", horizon=5)

    assert result is not None
    assert result['ticker'] == "PETR4.SA"
    assert result['model_type'] == "lstm"
    assert result['horizon'] == 5
    assert 'predictions' in result


@patch('src.ml.prediction.predictor.DataFetcher')
def test_predict_signal_with_mock_model(mock_fetcher_class, mock_data_fetcher):
    """Test signal prediction with mocked model."""
    mock_fetcher_class.return_value = mock_data_fetcher

    predictor = Predictor(use_cache=False)

    # Mock ensemble model
    mock_ensemble = Mock()
    mock_ensemble.predict.return_value = np.array([2])  # Buy signal
    mock_ensemble.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
    predictor.ensemble_model = mock_ensemble

    # Mock feature pipeline
    mock_df = Mock()
    mock_df.iloc.__getitem__.return_value = {'close': 30.0}
    predictor.feature_pipeline.prepare_features.return_value = mock_df

    result = predictor.predict_signal("PETR4.SA")

    assert result is not None
    assert result['ticker'] == "PETR4.SA"
    assert result['model_type'] == "ensemble"
    assert result['signal'] in ["buy", "sell", "hold"]
    assert 'confidence' in result
    assert 'probabilities' in result


def test_cache_integration():
    """Test cache integration."""
    with patch('src.ml.prediction.predictor.PredictionCache') as mock_cache_class:
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_cache_class.return_value = mock_cache

        predictor = Predictor(use_cache=True)

        assert predictor.cache is not None
        assert predictor.use_cache is True


def test_invalidate_cache():
    """Test cache invalidation."""
    with patch('src.ml.prediction.predictor.PredictionCache') as mock_cache_class:
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache

        predictor = Predictor(use_cache=True)
        predictor.invalidate_cache("PETR4.SA")

        mock_cache.invalidate_ticker.assert_called_once_with("PETR4.SA")


def test_get_cache_stats():
    """Test getting cache statistics."""
    with patch('src.ml.prediction.predictor.PredictionCache') as mock_cache_class:
        mock_cache = Mock()
        mock_cache.get_stats.return_value = {"hit_rate": 0.85, "total_keys": 42}
        mock_cache_class.return_value = mock_cache

        predictor = Predictor(use_cache=True)
        stats = predictor.get_cache_stats()

        assert stats["hit_rate"] == 0.85
        assert stats["total_keys"] == 42
