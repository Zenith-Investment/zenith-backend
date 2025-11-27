"""
Price forecasting module using advanced ML models and backtesting strategies.

Uses state-of-the-art machine learning:
- LSTM Neural Networks for time series
- XGBoost Gradient Boosting
- Random Forest Ensemble
- Ensemble combining all models

DISCLAIMER: Price predictions are for educational/informational purposes ONLY.
They are NOT investment recommendations. Past performance does not guarantee
future results. The user is solely responsible for their investment decisions.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional
import numpy as np
import pandas as pd
import structlog

from src.core.config import settings

logger = structlog.get_logger()


@dataclass
class PriceForecast:
    """Price forecast result."""
    ticker: str
    current_price: float
    forecast_date: date
    predicted_price: float
    predicted_change_pct: float
    confidence: float  # 0-1
    prediction_range: tuple[float, float]  # (low, high)
    methodology: str
    factors: list[str]
    strategy_backtests: list[dict]
    disclaimer: str = (
        "⚠️ AVISO IMPORTANTE: Esta previsão é baseada em análise de dados históricos "
        "e modelos estatísticos. NÃO é uma recomendação de investimento. "
        "O mercado é imprevisível e resultados passados NÃO garantem resultados futuros. "
        "A decisão de investir é de total responsabilidade do usuário. "
        "Consulte um profissional certificado antes de investir."
    )

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "current_price": round(self.current_price, 2),
            "forecast_date": self.forecast_date.isoformat(),
            "predicted_price": round(self.predicted_price, 2),
            "predicted_change_pct": round(self.predicted_change_pct, 2),
            "confidence": round(self.confidence, 2),
            "prediction_range": {
                "low": round(self.prediction_range[0], 2),
                "high": round(self.prediction_range[1], 2),
            },
            "methodology": self.methodology,
            "factors": self.factors,
            "strategy_backtests": self.strategy_backtests,
            "disclaimer": self.disclaimer,
            "user_decision_required": True,
        }


class PriceForecastEngine:
    """
    Engine for price forecasting using multiple methodologies.

    Note: This provides ESTIMATES based on historical data analysis.
    Users must make their own investment decisions.
    """

    def __init__(self):
        self.methods = [
            "moving_average",
            "linear_regression",
            "momentum",
            "mean_reversion",
        ]

    def forecast_with_moving_average(
        self,
        prices: pd.Series,
        periods: int = 20,
    ) -> tuple[float, float]:
        """Forecast using moving average trend."""
        if len(prices) < periods * 2:
            return prices.iloc[-1], 0.3

        # Calculate trend from MA
        ma = prices.rolling(window=periods).mean()
        current_ma = ma.iloc[-1]
        prev_ma = ma.iloc[-periods]

        trend = (current_ma - prev_ma) / prev_ma
        predicted = float(prices.iloc[-1] * (1 + trend))

        # Confidence based on trend consistency
        ma_diff = (prices - ma).dropna()
        volatility = ma_diff.std() / current_ma
        confidence = max(0.2, min(0.7, 0.7 - volatility))

        return predicted, confidence

    def forecast_with_linear_regression(
        self,
        prices: pd.Series,
        days_ahead: int = 30,
    ) -> tuple[float, float]:
        """Forecast using linear regression."""
        if len(prices) < 60:
            return prices.iloc[-1], 0.3

        # Simple linear regression
        x = np.arange(len(prices))
        y = prices.values

        # Calculate coefficients
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Predict
        future_x = len(prices) + days_ahead
        predicted = slope * future_x + intercept

        # R-squared for confidence
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        confidence = max(0.2, min(0.6, r_squared * 0.8))

        return float(predicted), confidence

    def forecast_with_momentum(
        self,
        prices: pd.Series,
        lookback: int = 20,
    ) -> tuple[float, float]:
        """Forecast using momentum analysis."""
        if len(prices) < lookback * 2:
            return prices.iloc[-1], 0.3

        # Calculate momentum
        current_price = prices.iloc[-1]
        past_price = prices.iloc[-lookback]
        momentum = (current_price - past_price) / past_price

        # Project forward (momentum tends to continue but with decay)
        decay_factor = 0.5
        predicted = float(current_price * (1 + momentum * decay_factor))

        # Confidence based on momentum consistency
        returns = prices.pct_change().dropna()[-lookback:]
        positive_days = (returns > 0).sum()
        consistency = positive_days / len(returns) if momentum > 0 else (len(returns) - positive_days) / len(returns)

        confidence = max(0.2, min(0.6, consistency * 0.7))

        return predicted, confidence

    def forecast_with_mean_reversion(
        self,
        prices: pd.Series,
        lookback: int = 60,
    ) -> tuple[float, float]:
        """Forecast using mean reversion."""
        if len(prices) < lookback:
            return prices.iloc[-1], 0.3

        current_price = prices.iloc[-1]
        mean_price = prices.iloc[-lookback:].mean()
        std_price = prices.iloc[-lookback:].std()

        # Z-score
        z_score = (current_price - mean_price) / std_price if std_price > 0 else 0

        # Mean reversion prediction
        reversion_strength = 0.3
        predicted = float(current_price + (mean_price - current_price) * reversion_strength)

        # Confidence higher when z-score is extreme
        confidence = max(0.2, min(0.6, abs(z_score) * 0.2))

        return predicted, confidence

    async def generate_forecast(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        forecast_days: int = 30,
        include_backtests: bool = True,
        use_ml: bool = True,
    ) -> PriceForecast:
        """
        Generate price forecast using advanced ML models and statistical methods.

        Args:
            ticker: Asset ticker
            price_data: DataFrame with OHLCV data
            forecast_days: Days ahead to forecast
            include_backtests: Include strategy backtest results
            use_ml: Use ML models for prediction (LSTM, XGBoost, RF)
        """
        prices = price_data["close"]
        current_price = float(prices.iloc[-1])
        forecast_date = date.today() + timedelta(days=forecast_days)

        # Get predictions from all methods
        predictions = []
        ml_prediction = None
        ml_features = None

        # Try ML prediction first (if enabled and enough data)
        if use_ml and len(price_data) >= 120:
            try:
                from src.ai.ml.model_trainer import model_trainer

                # Train/load model and predict
                ml_result = await model_trainer.predict(
                    ticker=ticker,
                    price_data=price_data,
                    target_days=min(forecast_days // 7, 10),  # Predict in weeks
                )

                ml_prediction = ml_result
                predictions.append((
                    "ML Ensemble (LSTM+XGBoost+RF)",
                    ml_result.predicted_price,
                    ml_result.confidence,
                ))

                ml_features = ml_result.feature_importance

                logger.info(
                    "ML prediction generated",
                    ticker=ticker,
                    predicted_price=ml_result.predicted_price,
                    confidence=ml_result.confidence,
                )

            except Exception as e:
                logger.warning(f"ML prediction failed, using fallback", error=str(e))

        # Statistical methods (always include as backup/ensemble)
        # Moving Average
        ma_pred, ma_conf = self.forecast_with_moving_average(prices)
        predictions.append(("Média Móvel", ma_pred, ma_conf))

        # Linear Regression
        lr_pred, lr_conf = self.forecast_with_linear_regression(prices, forecast_days)
        predictions.append(("Regressão Linear", lr_pred, lr_conf))

        # Momentum
        mom_pred, mom_conf = self.forecast_with_momentum(prices)
        predictions.append(("Momentum", mom_pred, mom_conf))

        # Mean Reversion
        mr_pred, mr_conf = self.forecast_with_mean_reversion(prices)
        predictions.append(("Reversão à Média", mr_pred, mr_conf))

        # Weighted average prediction (ML gets higher weight)
        weights = []
        for method, _, conf in predictions:
            if "ML Ensemble" in method:
                weights.append(conf * 2.0)  # Double weight for ML
            else:
                weights.append(conf)

        total_weight = sum(weights)
        if total_weight > 0:
            # Calculate weighted average of predictions
            # predictions format: (method_name, predicted_value, confidence)
            predicted_price = sum(pred * w for (_, pred, _), w in zip(predictions, weights)) / total_weight

            # Adjust confidence based on ML availability
            avg_confidence = sum(weights) / len(predictions)
            if ml_prediction:
                avg_confidence = min(0.85, avg_confidence * 1.2)
        else:
            predicted_price = current_price
            avg_confidence = 0.3

        # Calculate prediction range
        returns = prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(forecast_days)

        if ml_prediction:
            # Use ML prediction range if available
            range_low = ml_prediction.prediction_low
            range_high = ml_prediction.prediction_high
        else:
            range_low = predicted_price * (1 - volatility * 2)
            range_high = predicted_price * (1 + volatility * 2)

        change_pct = (predicted_price - current_price) / current_price * 100

        # Build factors list
        factors = [
            f"Análise de {len(prices)} dias de histórico",
            f"Volatilidade histórica: {returns.std() * 100:.2f}% ao dia",
        ]

        if ml_prediction:
            factors.insert(0, "✨ Previsão potencializada por Machine Learning (LSTM + XGBoost + Random Forest)")

        for method, pred, conf in predictions:
            direction = "alta" if pred > current_price else "baixa"
            factors.append(f"{method}: indica {direction} (confiança: {conf:.0%})")

        # Add top ML features if available
        if ml_features:
            top_features = sorted(ml_features.items(), key=lambda x: x[1], reverse=True)[:5]
            factors.append("Top fatores ML: " + ", ".join([f[0] for f in top_features]))

        # Strategy backtests
        strategy_backtests = []
        if include_backtests:
            strategy_backtests = await self._run_strategy_backtests(ticker, price_data)

        # Determine methodology description
        if ml_prediction:
            methodology = "ML Ensemble (LSTM + XGBoost + RF) + Análise Estatística"
        else:
            methodology = "Ensemble Estatístico (MA + Regressão + Momentum + Reversão)"

        return PriceForecast(
            ticker=ticker,
            current_price=current_price,
            forecast_date=forecast_date,
            predicted_price=predicted_price,
            predicted_change_pct=change_pct,
            confidence=avg_confidence,
            prediction_range=(range_low, range_high),
            methodology=methodology,
            factors=factors,
            strategy_backtests=strategy_backtests,
        )

    async def _run_strategy_backtests(
        self,
        ticker: str,
        price_data: pd.DataFrame,
    ) -> list[dict]:
        """Run strategy backtests for comparison."""
        from src.ai.backtesting import (
            BacktestEngine,
            BuyAndHoldStrategy,
            DCAStrategy,
            MomentumStrategy,
        )

        results = []
        engine = BacktestEngine(initial_capital=Decimal("10000"))

        # Use last 252 trading days (1 year) for backtest
        if len(price_data) < 252:
            return results

        test_data = price_data.iloc[-252:]
        start_date = test_data.index[0].date()
        end_date = test_data.index[-1].date()

        strategies = [
            BuyAndHoldStrategy(tickers=[ticker]),
            DCAStrategy(tickers=[ticker], investment_amount=Decimal("1000")),
            MomentumStrategy(tickers=[ticker], lookback_days=20),
        ]

        for strategy in strategies:
            try:
                result = await engine.run(
                    strategy=strategy,
                    price_data={ticker: test_data},
                    start_date=start_date,
                    end_date=end_date,
                )
                results.append({
                    "strategy": strategy.name,
                    "total_return": round(result.total_return, 2),
                    "annualized_return": round(result.annualized_return, 2),
                    "max_drawdown": round(result.max_drawdown, 2),
                    "sharpe_ratio": round(result.sharpe_ratio, 2),
                })
            except Exception as e:
                logger.error(f"Backtest failed for {strategy.name}", error=str(e))

        return results


class MiniMaxM2Integration:
    """
    Integration layer for MiniMax M2 model fine-tuning.

    This class provides the structure for fine-tuning MiniMax M2
    on financial data for improved predictions.
    """

    def __init__(self):
        self.model_id = "minimax-m2"
        self.fine_tuned = False
        self.api_endpoint = getattr(settings, "MINIMAX_API_ENDPOINT", None)
        self.api_key = getattr(settings, "MINIMAX_API_KEY", None)

    async def prepare_training_data(
        self,
        historical_data: dict[str, pd.DataFrame],
        market_events: list[dict],
    ) -> list[dict]:
        """
        Prepare training data for fine-tuning.

        Format: instruction-response pairs for financial analysis.
        """
        training_samples = []

        for ticker, df in historical_data.items():
            # Create samples from historical patterns
            for i in range(60, len(df) - 30, 30):
                past_data = df.iloc[i-60:i]
                future_data = df.iloc[i:i+30]

                past_return = (past_data["close"].iloc[-1] - past_data["close"].iloc[0]) / past_data["close"].iloc[0] * 100
                future_return = (future_data["close"].iloc[-1] - future_data["close"].iloc[0]) / future_data["close"].iloc[0] * 100

                # Don't train on actual predictions - train on analysis
                sample = {
                    "instruction": f"Analise o ativo {ticker} com base nos últimos 60 dias. "
                                  f"O retorno foi de {past_return:.2f}%. "
                                  "Quais fatores podem ter influenciado esse movimento?",
                    "response": self._generate_analysis_response(ticker, past_data, past_return),
                }
                training_samples.append(sample)

        return training_samples

    def _generate_analysis_response(
        self,
        ticker: str,
        data: pd.DataFrame,
        return_pct: float,
    ) -> str:
        """Generate educational analysis response for training."""
        volatility = data["close"].pct_change().std() * 100

        if return_pct > 10:
            trend = "forte alta"
        elif return_pct > 0:
            trend = "leve alta"
        elif return_pct > -10:
            trend = "leve baixa"
        else:
            trend = "forte baixa"

        return f"""
O ativo {ticker} apresentou {trend} de {return_pct:.2f}% no período analisado.

Análise técnica:
- Volatilidade diária média: {volatility:.2f}%
- Volume médio: variável ao longo do período

Fatores que podem ter influenciado:
- Condições gerais de mercado
- Notícias específicas do setor/empresa
- Fluxo de capital estrangeiro
- Indicadores macroeconômicos

Importante: Esta análise é baseada em dados históricos e tem fins educacionais.
Movimentos passados não garantem movimentos futuros. Para decisões de investimento,
consulte um profissional certificado e faça sua própria análise.
"""

    async def fine_tune(
        self,
        training_data: list[dict],
        validation_split: float = 0.1,
    ) -> dict:
        """
        Fine-tune MiniMax M2 model.

        Note: This requires MiniMax API access and appropriate credentials.
        """
        if not self.api_endpoint or not self.api_key:
            return {
                "status": "error",
                "message": "MiniMax API not configured. Add MINIMAX_API_ENDPOINT and MINIMAX_API_KEY to settings.",
            }

        # Split data
        split_idx = int(len(training_data) * (1 - validation_split))
        train_set = training_data[:split_idx]
        val_set = training_data[split_idx:]

        # In production, this would call the MiniMax fine-tuning API
        # For now, return structure for future implementation
        return {
            "status": "pending",
            "message": "Fine-tuning job prepared",
            "config": {
                "model": self.model_id,
                "training_samples": len(train_set),
                "validation_samples": len(val_set),
                "epochs": 3,
                "learning_rate": 2e-5,
            },
            "note": "Implement MiniMax API integration when credentials are available",
        }

    async def predict(
        self,
        ticker: str,
        context: str,
    ) -> dict:
        """
        Get prediction from fine-tuned model.

        Returns analysis (not price prediction) to avoid regulatory issues.
        """
        if not self.fine_tuned:
            return {
                "status": "not_fine_tuned",
                "message": "Model has not been fine-tuned yet",
            }

        # This would call the fine-tuned model
        # For now, return placeholder
        return {
            "status": "success",
            "analysis": f"Análise do {ticker} baseada no contexto fornecido...",
            "disclaimer": "Esta análise é gerada por IA e não constitui recomendação de investimento.",
        }


# Global instances
forecast_engine = PriceForecastEngine()
minimax_integration = MiniMaxM2Integration()
