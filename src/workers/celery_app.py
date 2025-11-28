from celery import Celery
from celery.schedules import crontab

from src.core.config import settings

celery_app = Celery(
    "investai_workers",
    broker=f"amqp://investai:investai@{settings.REDIS_HOST}:5672//",
    backend=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/1",
    include=[
        "src.workers.tasks.market_data",
        "src.workers.tasks.notifications",
        "src.workers.tasks.ai_processing",
        "src.workers.tasks.broker_sync",
        "src.workers.tasks.ml_training",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="America/Sao_Paulo",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Scheduled tasks (Celery Beat)
celery_app.conf.beat_schedule = {
    # Market data updates
    "update-market-prices": {
        "task": "src.workers.tasks.market_data.update_market_prices",
        "schedule": 300.0,  # Every 5 minutes
    },
    "update-market-indices": {
        "task": "src.workers.tasks.market_data.update_market_indices",
        "schedule": 60.0,  # Every minute
    },
    # Portfolio snapshots - daily at market close (18:00 São Paulo)
    "update-portfolio-snapshots": {
        "task": "src.workers.tasks.market_data.update_portfolio_snapshots",
        "schedule": crontab(hour=18, minute=30),
    },
    # Daily reports
    "process-daily-reports": {
        "task": "src.workers.tasks.notifications.send_daily_reports",
        "schedule": crontab(hour=19, minute=0),
    },
    # Check price alerts every 5 minutes
    "check-price-alerts": {
        "task": "src.workers.tasks.notifications.check_price_alerts",
        "schedule": 300.0,
    },
    # Broker sync - every 4 hours during market hours
    "sync-broker-connections": {
        "task": "src.workers.tasks.broker_sync.sync_all_broker_connections",
        "schedule": crontab(hour="10,14,18", minute=0),
    },
    # Check broker token expiry every hour
    "check-broker-token-expiry": {
        "task": "src.workers.tasks.broker_sync.check_broker_token_expiry",
        "schedule": crontab(minute=30),  # Every hour at :30
    },
    # ML model retraining - weekly on Sunday at 2 AM
    "retrain-ml-models-weekly": {
        "task": "src.workers.tasks.ml_training.retrain_all_models",
        "schedule": crontab(hour=2, minute=0, day_of_week=0),  # Sunday 2 AM
    },
    # Clean up old model versions monthly
    "cleanup-old-models-monthly": {
        "task": "src.workers.tasks.ml_training.cleanup_old_models",
        "schedule": crontab(hour=3, minute=0, day_of_month=1),  # 1st of month at 3 AM
    },
    # Clear prediction cache daily
    "clear-ml-cache-daily": {
        "task": "src.workers.tasks.ml_training.clear_prediction_cache",
        "schedule": crontab(hour=0, minute=0),  # Midnight
    },
}
