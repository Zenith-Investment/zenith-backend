"""
Celery tasks for ML model training and maintenance.
"""

from celery import shared_task
import logging

logger = logging.getLogger(__name__)


@shared_task(name="src.workers.tasks.ml_training.retrain_model", bind=True)
def retrain_model(self, ticker: str, model_type: str = "all", epochs: int = 100):
    """
    Retrain ML model for a specific ticker.

    Args:
        ticker: Stock ticker symbol
        model_type: Type of model (lstm, ensemble, or all)
        epochs: Number of epochs for LSTM training
    """
    from src.ml.training.trainer import ModelTrainer
    from src.ml.models.model_registry import ModelRegistry
    from src.ml.prediction.predictor import Predictor

    try:
        logger.info(f"Starting retraining for {ticker} - model_type: {model_type}")

        trainer = ModelTrainer(models_path="/app/models")
        registry = ModelRegistry(registry_path="/app/models/registry.json")
        predictor = Predictor(use_cache=True)

        results = {}

        # Train LSTM
        if model_type in ["lstm", "all"]:
            logger.info(f"Training LSTM for {ticker}")
            model, metrics, model_path = trainer.train_lstm_model(
                ticker=ticker,
                epochs=epochs,
                save_model=True
            )

            # Register model
            version_id = registry.register_model(
                model_type="lstm",
                model_path=model_path,
                ticker=ticker,
                metrics=metrics,
                metadata={"auto_retrained": True, "task_id": self.request.id}
            )

            results["lstm"] = {
                "status": "success",
                "version_id": version_id,
                "metrics": metrics
            }

        # Train Ensemble
        if model_type in ["ensemble", "all"]:
            logger.info(f"Training ensemble for {ticker}")
            model, metrics, model_path = trainer.train_ensemble_model(
                ticker=ticker,
                save_model=True
            )

            # Register model
            version_id = registry.register_model(
                model_type="ensemble",
                model_path=model_path,
                ticker=ticker,
                metrics=metrics,
                metadata={"auto_retrained": True, "task_id": self.request.id}
            )

            results["ensemble"] = {
                "status": "success",
                "version_id": version_id,
                "metrics": metrics
            }

        # Invalidate cache
        predictor.invalidate_cache(ticker)

        logger.info(f"Retraining completed for {ticker}: {results}")

        return {
            "ticker": ticker,
            "status": "success",
            "results": results
        }

    except Exception as e:
        logger.error(f"Retraining failed for {ticker}: {e}")
        return {
            "ticker": ticker,
            "status": "failed",
            "error": str(e)
        }


@shared_task(name="src.workers.tasks.ml_training.retrain_all_models")
def retrain_all_models(tickers: list = None, model_type: str = "all", epochs: int = 100):
    """
    Retrain models for multiple tickers.

    Args:
        tickers: List of stock ticker symbols (defaults to popular Brazilian stocks)
        model_type: Type of model to retrain
        epochs: Number of epochs for LSTM
    """
    if tickers is None:
        tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA"]

    logger.info(f"Starting batch retraining for {len(tickers)} tickers")

    results = []

    for ticker in tickers:
        result = retrain_model.delay(ticker, model_type, epochs)
        results.append({
            "ticker": ticker,
            "task_id": result.id
        })

    return {
        "status": "queued",
        "total_tickers": len(tickers),
        "tasks": results
    }


@shared_task(name="src.workers.tasks.ml_training.cleanup_old_models")
def cleanup_old_models(keep_latest_n: int = 5, keep_best: bool = True):
    """
    Clean up old model versions.

    Args:
        keep_latest_n: Number of latest versions to keep
        keep_best: Whether to always keep the best performing model
    """
    from src.ml.models.model_registry import ModelRegistry

    try:
        logger.info(f"Starting model cleanup (keep latest {keep_latest_n}, keep best: {keep_best})")

        registry = ModelRegistry(registry_path="/app/models/registry.json")
        registry.cleanup_old_versions(keep_latest_n=keep_latest_n, keep_best=keep_best)

        stats = registry.get_stats()

        logger.info(f"Model cleanup completed: {stats}")

        return {
            "status": "success",
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Model cleanup failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


@shared_task(name="src.workers.tasks.ml_training.clear_prediction_cache")
def clear_prediction_cache():
    """Clear all prediction cache."""
    from src.ml.prediction.predictor import Predictor

    try:
        logger.info("Clearing prediction cache")

        predictor = Predictor(use_cache=True)
        predictor.invalidate_cache()

        stats = predictor.get_cache_stats()

        logger.info(f"Cache cleared: {stats}")

        return {
            "status": "success",
            "cache_stats": stats
        }

    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


@shared_task(name="src.workers.tasks.ml_training.evaluate_model")
def evaluate_model(ticker: str, model_type: str):
    """
    Evaluate model performance on recent data.

    Args:
        ticker: Stock ticker symbol
        model_type: Type of model (lstm or ensemble)
    """
    from src.ml.prediction.predictor import Predictor
    from src.ml.training.evaluation import ModelEvaluator

    try:
        logger.info(f"Evaluating {model_type} model for {ticker}")

        predictor = Predictor()
        evaluator = ModelEvaluator()

        # TODO: Implement evaluation logic
        # Load latest model and evaluate on recent data

        return {
            "ticker": ticker,
            "model_type": model_type,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Model evaluation failed for {ticker}: {e}")
        return {
            "ticker": ticker,
            "model_type": model_type,
            "status": "failed",
            "error": str(e)
        }
