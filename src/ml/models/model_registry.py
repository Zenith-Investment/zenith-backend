"""
Model registry for versioning and management of ML models.
"""

import os
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing ML model versions.

    Keeps track of trained models, their performance metrics,
    and metadata for model lifecycle management.
    """

    def __init__(self, registry_path: str = "./models/registry.json"):
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = registry_path
        self.registry: Dict[str, List[Dict[str, Any]]] = {}

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)

        # Load existing registry
        self.load()

    def load(self):
        """Load registry from file."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.registry = json.load(f)
                logger.info(f"Loaded registry with {len(self.registry)} model types")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                self.registry = {}
        else:
            logger.info("No existing registry found, creating new one")
            self.registry = {}

    def save(self):
        """Save registry to file."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
            logger.info(f"Registry saved to {self.registry_path}")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def register_model(
        self,
        model_type: str,
        model_path: str,
        ticker: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new model version.

        Args:
            model_type: Type of model (lstm, ensemble, etc.)
            model_path: Path to saved model
            ticker: Stock ticker
            metrics: Performance metrics
            metadata: Additional metadata

        Returns:
            Version ID of registered model
        """
        # Generate version ID
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model entry
        model_entry = {
            "version_id": version_id,
            "model_type": model_type,
            "ticker": ticker,
            "model_path": model_path,
            "metrics": metrics,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "status": "active"
        }

        # Add to registry
        key = f"{model_type}_{ticker}"
        if key not in self.registry:
            self.registry[key] = []

        self.registry[key].append(model_entry)

        # Save registry
        self.save()

        logger.info(f"Registered {model_type} model for {ticker} (version: {version_id})")

        return version_id

    def get_latest_model(
        self,
        model_type: str,
        ticker: str,
        status: str = "active"
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest model version.

        Args:
            model_type: Type of model
            ticker: Stock ticker
            status: Model status filter

        Returns:
            Model entry or None if not found
        """
        key = f"{model_type}_{ticker}"

        if key not in self.registry:
            return None

        # Filter by status and sort by registration date
        models = [
            m for m in self.registry[key]
            if m.get("status") == status
        ]

        if not models:
            return None

        # Return latest
        return max(models, key=lambda x: x["registered_at"])

    def get_best_model(
        self,
        model_type: str,
        ticker: str,
        metric: str = "accuracy",
        status: str = "active"
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best performing model based on a metric.

        Args:
            model_type: Type of model
            ticker: Stock ticker
            metric: Metric to optimize
            status: Model status filter

        Returns:
            Model entry or None if not found
        """
        key = f"{model_type}_{ticker}"

        if key not in self.registry:
            return None

        # Filter by status
        models = [
            m for m in self.registry[key]
            if m.get("status") == status and metric in m.get("metrics", {})
        ]

        if not models:
            return None

        # Return best (higher is better for most metrics)
        # For loss metrics like MSE, you might want to use min
        if metric in ["mse", "rmse", "mae", "loss"]:
            return min(models, key=lambda x: x["metrics"][metric])
        else:
            return max(models, key=lambda x: x["metrics"][metric])

    def get_model_history(
        self,
        model_type: str,
        ticker: str
    ) -> List[Dict[str, Any]]:
        """
        Get all model versions for a ticker.

        Args:
            model_type: Type of model
            ticker: Stock ticker

        Returns:
            List of model entries
        """
        key = f"{model_type}_{ticker}"
        return self.registry.get(key, [])

    def deactivate_model(
        self,
        model_type: str,
        ticker: str,
        version_id: str
    ) -> bool:
        """
        Deactivate a specific model version.

        Args:
            model_type: Type of model
            ticker: Stock ticker
            version_id: Version ID to deactivate

        Returns:
            True if successful, False otherwise
        """
        key = f"{model_type}_{ticker}"

        if key not in self.registry:
            return False

        for model in self.registry[key]:
            if model["version_id"] == version_id:
                model["status"] = "inactive"
                model["deactivated_at"] = datetime.now().isoformat()
                self.save()
                logger.info(f"Deactivated {model_type} model {version_id} for {ticker}")
                return True

        return False

    def cleanup_old_versions(
        self,
        keep_latest_n: int = 5,
        keep_best: bool = True
    ):
        """
        Clean up old model versions.

        Args:
            keep_latest_n: Number of latest versions to keep
            keep_best: Whether to always keep the best performing model
        """
        for key in self.registry:
            models = self.registry[key]

            if len(models) <= keep_latest_n:
                continue

            # Sort by date
            sorted_models = sorted(models, key=lambda x: x["registered_at"], reverse=True)

            # Keep latest N
            to_keep = set(m["version_id"] for m in sorted_models[:keep_latest_n])

            # Keep best if requested
            if keep_best and models:
                # Find best based on first available metric
                for metric in ["accuracy", "f1_score", "r2", "rmse"]:
                    models_with_metric = [m for m in models if metric in m.get("metrics", {})]
                    if models_with_metric:
                        if metric in ["rmse", "mse", "mae"]:
                            best = min(models_with_metric, key=lambda x: x["metrics"][metric])
                        else:
                            best = max(models_with_metric, key=lambda x: x["metrics"][metric])
                        to_keep.add(best["version_id"])
                        break

            # Mark others as inactive
            for model in models:
                if model["version_id"] not in to_keep and model.get("status") == "active":
                    model["status"] = "archived"
                    model["archived_at"] = datetime.now().isoformat()

        self.save()
        logger.info("Cleaned up old model versions")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with stats
        """
        total_models = sum(len(models) for models in self.registry.values())
        active_models = sum(
            len([m for m in models if m.get("status") == "active"])
            for models in self.registry.values()
        )

        model_types = {}
        for key, models in self.registry.items():
            model_type = key.split('_')[0]
            model_types[model_type] = model_types.get(model_type, 0) + len(models)

        return {
            "total_models": total_models,
            "active_models": active_models,
            "model_types": model_types,
            "tickers": list(set(key.split('_')[1] for key in self.registry.keys()))
        }
