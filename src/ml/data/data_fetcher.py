"""
Data fetcher for historical stock prices.

Fetches data from Yahoo Finance and other sources.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch historical stock price data from various sources."""

    @staticmethod
    def fetch_yfinance(
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'PETR4.SA')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch (if start_date/end_date not provided)
                    Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: Data interval
                      Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching data for {ticker} from Yahoo Finance")

            stock = yf.Ticker(ticker)

            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date, interval=interval)
            else:
                df = stock.history(period=period, interval=interval)

            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")

            # Reset index to make date a column
            df = df.reset_index()

            # Rename columns to lowercase (after reset_index to include the index column)
            df.columns = df.columns.str.lower()

            # Rename 'date' or 'datetime' to 'timestamp'
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            elif 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})
            elif df.columns[0].lower() in ['date', 'datetime']:
                # If the first column is the date/datetime column
                df = df.rename(columns={df.columns[0]: 'timestamp'})

            # Select only OHLCV columns
            columns_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in columns_to_keep if col in df.columns]]

            # Set timestamp as index
            df = df.set_index('timestamp')

            logger.info(f"Fetched {len(df)} rows for {ticker}")

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            raise

    @staticmethod
    def fetch_multiple_tickers(
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d"
    ) -> dict:
        """
        Fetch historical data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch
            interval: Data interval

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}

        for ticker in tickers:
            try:
                df = DataFetcher.fetch_yfinance(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    period=period,
                    interval=interval
                )
                results[ticker] = df
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                results[ticker] = None

        return results

    @staticmethod
    def get_ticker_info(ticker: str) -> dict:
        """
        Get ticker information and metadata.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with ticker information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                "symbol": ticker,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "currency": info.get("currency", ""),
                "exchange": info.get("exchange", ""),
            }

        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {e}")
            return {}

    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """
        Validate if a ticker symbol exists.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if ticker is valid, False otherwise
        """
        try:
            stock = yf.Ticker(ticker)
            # Try to fetch 1 day of data
            df = stock.history(period="1d")
            return not df.empty
        except Exception:
            return False

    @staticmethod
    def get_latest_price(ticker: str) -> Optional[float]:
        """
        Get the latest price for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Latest close price or None if not available
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1d")

            if df.empty:
                return None

            return float(df['Close'].iloc[-1])

        except Exception as e:
            logger.error(f"Error getting latest price for {ticker}: {e}")
            return None
