"""
Unit tests for ML Models.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch


def test_lstm_model_initialization():
    """Test LSTM model initialization."""
    from src.ml.models.lstm_model import LSTMModel

    model = LSTMModel(
        sequence_length=60,
        n_features=10,
        lstm_units=[50, 50],
        dropout_rate=0.2
    )

    assert model.sequence_length == 60
    assert model.n_features == 10
    assert model.lstm_units == [50, 50]
    assert model.dropout_rate == 0.2
    assert model.model is None


def test_lstm_model_build():
    """Test LSTM model building."""
    from src.ml.models.lstm_model import LSTMModel

    model = LSTMModel(sequence_length=10, n_features=5)
    model.build_model()

    assert model.model is not None
    assert len(model.model.layers) > 0


def test_lstm_model_train():
    """Test LSTM model training."""
    from src.ml.models.lstm_model import LSTMModel

    model = LSTMModel(sequence_length=10, n_features=5)

    # Create dummy data
    X_train = np.random.randn(100, 10, 5)
    y_train = np.random.randn(100, 1)
    X_val = np.random.randn(20, 10, 5)
    y_val = np.random.randn(20, 1)

    history = model.train(X_train, y_train, X_val, y_val, epochs=1, verbose=0)

    assert history is not None
    assert 'loss' in history.history


def test_lstm_model_predict():
    """Test LSTM model prediction."""
    from src.ml.models.lstm_model import LSTMModel

    model = LSTMModel(sequence_length=10, n_features=5)

    # Train with minimal data
    X_train = np.random.randn(50, 10, 5)
    y_train = np.random.randn(50, 1)
    model.train(X_train, y_train, epochs=1, verbose=0)

    # Predict
    last_sequence = np.random.randn(10, 5)
    predictions = model.predict_next(last_sequence, n_steps=5)

    assert predictions.shape == (5,)


def test_lstm_model_save_load():
    """Test LSTM model save and load."""
    from src.ml.models.lstm_model import LSTMModel

    model = LSTMModel(sequence_length=10, n_features=5)

    # Train with minimal data
    X_train = np.random.randn(50, 10, 5)
    y_train = np.random.randn(50, 1)
    model.train(X_train, y_train, epochs=1, verbose=0)

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_lstm")
        model.save(save_path)

        new_model = LSTMModel(sequence_length=10, n_features=5)
        new_model.load(save_path)

        assert new_model.model is not None


def test_ensemble_model_initialization():
    """Test ensemble model initialization."""
    from src.ml.models.ensemble_model import EnsembleModel

    model = EnsembleModel(task="classification")

    assert model.task == "classification"
    assert model.rf_model is not None
    assert model.xgb_model is not None


def test_ensemble_model_train_classification():
    """Test ensemble model training for classification."""
    from src.ml.models.ensemble_model import EnsembleModel

    model = EnsembleModel(task="classification")

    # Create dummy data
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 3, 100)
    X_val = np.random.randn(20, 10)
    y_val = np.random.randint(0, 3, 20)

    model.train(X_train, y_train, X_val, y_val)

    assert model.rf_model is not None
    assert model.xgb_model is not None


def test_ensemble_model_predict_classification():
    """Test ensemble model prediction for classification."""
    from src.ml.models.ensemble_model import EnsembleModel

    model = EnsembleModel(task="classification")

    # Train with minimal data
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 3, 100)
    model.train(X_train, y_train)

    # Predict
    X_test = np.random.randn(10, 10)
    predictions = model.predict(X_test)

    assert predictions.shape == (10,)
    assert np.all((predictions >= 0) & (predictions <= 2))


def test_ensemble_model_predict_proba():
    """Test ensemble model probability prediction."""
    from src.ml.models.ensemble_model import EnsembleModel

    model = EnsembleModel(task="classification")

    # Train with minimal data
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 3, 100)
    model.train(X_train, y_train)

    # Predict probabilities
    X_test = np.random.randn(10, 10)
    probas = model.predict_proba(X_test)

    assert probas.shape == (10, 3)
    assert np.allclose(probas.sum(axis=1), 1.0, atol=0.01)


def test_ensemble_model_feature_importance():
    """Test ensemble model feature importance."""
    from src.ml.models.ensemble_model import EnsembleModel

    model = EnsembleModel(task="classification")

    # Train with minimal data
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 3, 100)
    model.train(X_train, y_train)

    # Get feature importance
    importance = model.get_feature_importance()

    assert importance is not None
    assert 'rf' in importance
    assert 'xgb' in importance
    assert len(importance['rf']) == 10
    assert len(importance['xgb']) == 10


def test_model_registry_initialization():
    """Test model registry initialization."""
    from src.ml.models.model_registry import ModelRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "registry.json")
        registry = ModelRegistry(registry_path=registry_path)

        assert registry.registry_path == registry_path
        assert isinstance(registry.registry, dict)


def test_model_registry_register():
    """Test model registration."""
    from src.ml.models.model_registry import ModelRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "registry.json")
        registry = ModelRegistry(registry_path=registry_path)

        version_id = registry.register_model(
            model_type="lstm",
            model_path="/fake/path",
            ticker="PETR4.SA",
            metrics={"mse": 0.01, "mae": 0.05}
        )

        assert version_id is not None
        assert "lstm_PETR4.SA" in registry.registry
        assert len(registry.registry["lstm_PETR4.SA"]) == 1


def test_model_registry_get_latest():
    """Test getting latest model."""
    from src.ml.models.model_registry import ModelRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "registry.json")
        registry = ModelRegistry(registry_path=registry_path)

        # Register multiple models
        registry.register_model("lstm", "/path1", "PETR4.SA", {"mse": 0.01})
        registry.register_model("lstm", "/path2", "PETR4.SA", {"mse": 0.02})

        latest = registry.get_latest_model("lstm", "PETR4.SA")

        assert latest is not None
        assert latest["model_path"] == "/path2"


def test_model_registry_get_best():
    """Test getting best model."""
    from src.ml.models.model_registry import ModelRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "registry.json")
        registry = ModelRegistry(registry_path=registry_path)

        # Register models with different metrics
        registry.register_model("lstm", "/path1", "PETR4.SA", {"mse": 0.02})
        registry.register_model("lstm", "/path2", "PETR4.SA", {"mse": 0.01})

        best = registry.get_best_model("lstm", "PETR4.SA", metric="mse")

        assert best is not None
        assert best["model_path"] == "/path2"
        assert best["metrics"]["mse"] == 0.01


def test_model_registry_stats():
    """Test getting registry statistics."""
    from src.ml.models.model_registry import ModelRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "registry.json")
        registry = ModelRegistry(registry_path=registry_path)

        registry.register_model("lstm", "/path1", "PETR4.SA", {"mse": 0.01})
        registry.register_model("ensemble", "/path2", "VALE3.SA", {"accuracy": 0.75})

        stats = registry.get_stats()

        assert stats["total_models"] == 2
        assert stats["active_models"] == 2
        assert "lstm" in stats["model_types"]
        assert "ensemble" in stats["model_types"]
