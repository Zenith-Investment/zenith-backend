"""
Model evaluation and backtesting utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and backtest ML models."""

    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)

        Returns:
            Dictionary with metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            roc_auc_score
        )

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # ROC AUC for binary classification
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            except Exception:
                pass

        return metrics

    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        # Direction accuracy
        if len(y_true) > 1:
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            direction_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
        else:
            direction_accuracy = 0.0

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "direction_accuracy": float(direction_accuracy)
        }

    @staticmethod
    def backtest_signals(
        df: pd.DataFrame,
        signals: np.ndarray,
        initial_capital: float = 10000.0,
        commission: float = 0.001
    ) -> Dict[str, Any]:
        """
        Backtest trading signals.

        Args:
            df: DataFrame with OHLCV data
            signals: Trading signals (-1=sell, 0=hold, 1=buy)
            initial_capital: Initial capital for backtesting
            commission: Trading commission (as fraction)

        Returns:
            Dictionary with backtest results
        """
        # Ensure signals and df have same length
        signals = signals[:len(df)]

        # Initialize
        capital = initial_capital
        position = 0  # 0=no position, 1=long
        shares = 0
        trades = []
        portfolio_values = []

        for i in range(len(df)):
            price = df['close'].iloc[i]
            signal = signals[i]

            # Buy signal
            if signal == 1 and position == 0:
                # Buy with all capital
                cost = capital * (1 + commission)
                if cost > 0:
                    shares = capital / cost
                    position = 1
                    trades.append({
                        "date": df.index[i],
                        "action": "buy",
                        "price": price,
                        "shares": shares,
                        "capital": capital
                    })
                    capital = 0

            # Sell signal
            elif signal == -1 and position == 1:
                # Sell all shares
                capital = shares * price * (1 - commission)
                trades.append({
                    "date": df.index[i],
                    "action": "sell",
                    "price": price,
                    "shares": shares,
                    "capital": capital
                })
                shares = 0
                position = 0

            # Calculate portfolio value
            if position == 1:
                portfolio_value = shares * price
            else:
                portfolio_value = capital

            portfolio_values.append(portfolio_value)

        # Final portfolio value
        if position == 1:
            # Sell remaining position
            final_capital = shares * df['close'].iloc[-1] * (1 - commission)
        else:
            final_capital = capital

        # Calculate metrics
        returns = (final_capital - initial_capital) / initial_capital * 100
        n_trades = len(trades)

        # Buy and hold comparison
        buy_hold_shares = initial_capital / df['close'].iloc[0]
        buy_hold_value = buy_hold_shares * df['close'].iloc[-1]
        buy_hold_return = (buy_hold_value - initial_capital) / initial_capital * 100

        # Calculate Sharpe ratio (simplified)
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-10) * np.sqrt(252)

        # Maximum drawdown
        cumulative = np.array(portfolio_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100

        return {
            "initial_capital": initial_capital,
            "final_capital": float(final_capital),
            "total_return_pct": float(returns),
            "buy_hold_return_pct": float(buy_hold_return),
            "n_trades": n_trades,
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown_pct": float(max_drawdown),
            "trades": trades,
            "portfolio_values": portfolio_values
        }

    @staticmethod
    def cross_validate_predictions(
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Perform time series cross-validation.

        Args:
            model: Model with fit/predict methods
            X: Features
            y: Targets
            n_splits: Number of CV splits

        Returns:
            Dictionary with CV results
        """
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Score
            score = np.mean((y_test - y_pred) ** 2)  # MSE
            scores.append(score)

        return {
            "cv_scores": scores,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "n_splits": n_splits
        }

    @staticmethod
    def generate_evaluation_report(
        model_name: str,
        metrics: Dict[str, float],
        backtest_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate evaluation report.

        Args:
            model_name: Name of the model
            metrics: Model metrics
            backtest_results: Backtesting results (optional)

        Returns:
            Formatted report string
        """
        report = f"""
{'='*60}
Model Evaluation Report: {model_name}
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE METRICS
{'='*60}
"""

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                report += f"{key:30s}: {value:.4f}\n"

        if backtest_results:
            report += f"""
BACKTEST RESULTS
{'='*60}
Initial Capital          : ${backtest_results['initial_capital']:,.2f}
Final Capital            : ${backtest_results['final_capital']:,.2f}
Total Return             : {backtest_results['total_return_pct']:.2f}%
Buy & Hold Return        : {backtest_results['buy_hold_return_pct']:.2f}%
Number of Trades         : {backtest_results['n_trades']}
Sharpe Ratio             : {backtest_results['sharpe_ratio']:.4f}
Maximum Drawdown         : {backtest_results['max_drawdown_pct']:.2f}%
"""

        report += f"\n{'='*60}\n"

        return report
