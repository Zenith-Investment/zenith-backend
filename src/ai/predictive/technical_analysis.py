"""Technical analysis module for price pattern analysis."""
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class TechnicalSignal:
    """Technical analysis signal."""
    indicator: str
    signal: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-100
    value: float
    description: str


@dataclass
class TechnicalAnalysisResult:
    """Complete technical analysis result."""
    ticker: str
    signals: list[TechnicalSignal]
    overall_signal: str  # "bullish", "bearish", "neutral"
    overall_strength: float
    support_levels: list[float]
    resistance_levels: list[float]
    trend: str  # "uptrend", "downtrend", "sideways"
    disclaimer: str = (
        "Análise técnica é baseada em dados passados e NÃO prevê o futuro. "
        "Use apenas como ferramenta auxiliar. Consulte um profissional certificado."
    )

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "signals": [
                {
                    "indicator": s.indicator,
                    "signal": s.signal,
                    "strength": s.strength,
                    "value": s.value,
                    "description": s.description,
                }
                for s in self.signals
            ],
            "overall_signal": self.overall_signal,
            "overall_strength": self.overall_strength,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "trend": self.trend,
            "disclaimer": self.disclaimer,
        }


class TechnicalAnalyzer:
    """Technical analysis calculator."""

    def __init__(self):
        pass

    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period).mean()

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = self.calculate_sma(prices, period)
        std = prices.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return upper_band, sma, lower_band

    def find_support_resistance(
        self,
        prices: pd.Series,
        window: int = 20,
        num_levels: int = 3,
    ) -> tuple[list[float], list[float]]:
        """Find support and resistance levels."""
        # Find local minima (support) and maxima (resistance)
        prices_array = prices.values

        # Simple approach: find peaks and troughs
        supports = []
        resistances = []

        for i in range(window, len(prices_array) - window):
            # Check if local minimum (support)
            if all(prices_array[i] <= prices_array[i-window:i]) and \
               all(prices_array[i] <= prices_array[i+1:i+window+1]):
                supports.append(float(prices_array[i]))

            # Check if local maximum (resistance)
            if all(prices_array[i] >= prices_array[i-window:i]) and \
               all(prices_array[i] >= prices_array[i+1:i+window+1]):
                resistances.append(float(prices_array[i]))

        # Get most recent levels
        supports = sorted(set(supports))[-num_levels:]
        resistances = sorted(set(resistances))[-num_levels:]

        return supports, resistances

    def determine_trend(
        self,
        prices: pd.Series,
        short_period: int = 20,
        long_period: int = 50,
    ) -> str:
        """Determine current trend using moving averages."""
        if len(prices) < long_period:
            return "insufficient_data"

        sma_short = self.calculate_sma(prices, short_period).iloc[-1]
        sma_long = self.calculate_sma(prices, long_period).iloc[-1]
        current_price = prices.iloc[-1]

        if current_price > sma_short > sma_long:
            return "uptrend"
        elif current_price < sma_short < sma_long:
            return "downtrend"
        else:
            return "sideways"

    def analyze(self, ticker: str, df: pd.DataFrame) -> TechnicalAnalysisResult:
        """
        Perform complete technical analysis on price data.

        Args:
            ticker: Asset ticker
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        """
        prices = df["close"]
        signals = []

        # RSI Analysis
        rsi = self.calculate_rsi(prices)
        current_rsi = rsi.iloc[-1]

        if current_rsi < 30:
            rsi_signal = TechnicalSignal(
                indicator="RSI",
                signal="bullish",
                strength=min(100, (30 - current_rsi) * 3),
                value=current_rsi,
                description=f"RSI em {current_rsi:.1f} indica sobrevenda (possível reversão)",
            )
        elif current_rsi > 70:
            rsi_signal = TechnicalSignal(
                indicator="RSI",
                signal="bearish",
                strength=min(100, (current_rsi - 70) * 3),
                value=current_rsi,
                description=f"RSI em {current_rsi:.1f} indica sobrecompra (possível correção)",
            )
        else:
            rsi_signal = TechnicalSignal(
                indicator="RSI",
                signal="neutral",
                strength=50,
                value=current_rsi,
                description=f"RSI em {current_rsi:.1f} em zona neutra",
            )
        signals.append(rsi_signal)

        # MACD Analysis
        macd_line, signal_line, histogram = self.calculate_macd(prices)
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2] if len(histogram) > 1 else 0
        current_price = prices.iloc[-1]

        # Normalize histogram by current price for consistent 0-100 scale
        # This prevents huge values for high-priced stocks
        hist_percent = (abs(current_hist) / current_price) * 100 if current_price > 0 else 0

        if current_hist > 0 and current_hist > prev_hist:
            macd_signal = TechnicalSignal(
                indicator="MACD",
                signal="bullish",
                strength=min(100, hist_percent * 20),  # Scale to 0-100 range
                value=current_hist,
                description="MACD com histograma positivo crescente (momentum de alta)",
            )
        elif current_hist < 0 and current_hist < prev_hist:
            macd_signal = TechnicalSignal(
                indicator="MACD",
                signal="bearish",
                strength=min(100, hist_percent * 20),  # Scale to 0-100 range
                value=current_hist,
                description="MACD com histograma negativo decrescente (momentum de baixa)",
            )
        else:
            macd_signal = TechnicalSignal(
                indicator="MACD",
                signal="neutral",
                strength=30,
                value=current_hist,
                description="MACD em transição",
            )
        signals.append(macd_signal)

        # Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(prices)
        current_price = prices.iloc[-1]

        if current_price < lower.iloc[-1]:
            bb_signal = TechnicalSignal(
                indicator="Bollinger Bands",
                signal="bullish",
                strength=70,
                value=current_price,
                description="Preço abaixo da banda inferior (possível sobrevenda)",
            )
        elif current_price > upper.iloc[-1]:
            bb_signal = TechnicalSignal(
                indicator="Bollinger Bands",
                signal="bearish",
                strength=70,
                value=current_price,
                description="Preço acima da banda superior (possível sobrecompra)",
            )
        else:
            bb_signal = TechnicalSignal(
                indicator="Bollinger Bands",
                signal="neutral",
                strength=40,
                value=current_price,
                description="Preço dentro das bandas de Bollinger",
            )
        signals.append(bb_signal)

        # Moving Averages
        sma20 = self.calculate_sma(prices, 20).iloc[-1]
        sma50 = self.calculate_sma(prices, 50).iloc[-1] if len(prices) >= 50 else sma20

        if current_price > sma20 > sma50:
            ma_signal = TechnicalSignal(
                indicator="Médias Móveis",
                signal="bullish",
                strength=60,
                value=sma20,
                description="Preço acima das médias (tendência de alta)",
            )
        elif current_price < sma20 < sma50:
            ma_signal = TechnicalSignal(
                indicator="Médias Móveis",
                signal="bearish",
                strength=60,
                value=sma20,
                description="Preço abaixo das médias (tendência de baixa)",
            )
        else:
            ma_signal = TechnicalSignal(
                indicator="Médias Móveis",
                signal="neutral",
                strength=40,
                value=sma20,
                description="Médias móveis sem direção clara",
            )
        signals.append(ma_signal)

        # Calculate overall signal
        bullish_count = sum(1 for s in signals if s.signal == "bullish")
        bearish_count = sum(1 for s in signals if s.signal == "bearish")

        if bullish_count > bearish_count:
            overall = "bullish"
            strength = sum(s.strength for s in signals if s.signal == "bullish") / bullish_count
        elif bearish_count > bullish_count:
            overall = "bearish"
            strength = sum(s.strength for s in signals if s.signal == "bearish") / bearish_count
        else:
            overall = "neutral"
            strength = 50

        # Support and resistance
        supports, resistances = self.find_support_resistance(prices)

        # Trend
        trend = self.determine_trend(prices)

        return TechnicalAnalysisResult(
            ticker=ticker,
            signals=signals,
            overall_signal=overall,
            overall_strength=strength,
            support_levels=supports,
            resistance_levels=resistances,
            trend=trend,
        )
