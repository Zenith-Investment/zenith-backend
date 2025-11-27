"""
MCP Client for integrating with external MCP servers and local tools.

Supports:
- Yahoo Finance (local, no API key needed)
- Alpha Vantage MCP
- BRAPI (Brazilian stocks)
- Custom financial tools
"""
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional
import httpx
import structlog

from src.core.config import settings
from src.mcp.tools import tool_registry, ToolCategory

logger = structlog.get_logger()


class MCPClient:
    """
    MCP Client for financial data and tools.

    Integrates multiple data sources:
    1. Yahoo Finance (via yfinance) - Free, no API key
    2. BRAPI - Brazilian stock market data
    3. Alpha Vantage - Global market data
    """

    def __init__(self):
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register tool handlers."""
        handlers = {
            "get_stock_quote": self.get_stock_quote,
            "get_stock_history": self.get_stock_history,
            "get_market_indices": self.get_market_indices,
            "search_stocks": self.search_stocks,
            "calculate_technical_indicators": self.calculate_technical_indicators,
            "get_company_fundamentals": self.get_company_fundamentals,
            "get_dividends": self.get_dividends,
            "get_market_news": self.get_market_news,
            "get_selic_rate": self.get_selic_rate,
            "get_inflation_data": self.get_inflation_data,
            "compare_stocks": self.compare_stocks,
        }

        for name, handler in handlers.items():
            tool = tool_registry.get(name)
            if tool:
                tool.handler = handler

    def _get_cache(self, key: str) -> Any | None:
        """Get cached value if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        """Set cache value."""
        self._cache[key] = (value, datetime.now())

    async def get_stock_quote(self, ticker: str) -> dict:
        """Get real-time stock quote."""
        ticker = ticker.upper().strip()

        # Check cache
        cache_key = f"quote:{ticker}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        # Try BRAPI first (Brazilian stocks)
        if settings.BRAPI_TOKEN:
            try:
                result = await self._get_brapi_quote(ticker)
                if result:
                    self._set_cache(cache_key, result)
                    return result
            except Exception as e:
                logger.warning("BRAPI quote failed", ticker=ticker, error=str(e))

        # Fallback to Yahoo Finance
        try:
            result = await self._get_yfinance_quote(ticker)
            if result:
                self._set_cache(cache_key, result)
                return result
        except Exception as e:
            logger.warning("YFinance quote failed", ticker=ticker, error=str(e))

        return {"error": f"Nao foi possivel obter cotacao para {ticker}"}

    async def _get_brapi_quote(self, ticker: str) -> dict | None:
        """Get quote from BRAPI."""
        url = f"https://brapi.dev/api/quote/{ticker}"
        params = {"token": settings.BRAPI_TOKEN}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    result = data["results"][0]
                    return {
                        "ticker": ticker,
                        "name": result.get("longName", result.get("shortName", "")),
                        "price": result.get("regularMarketPrice"),
                        "change": result.get("regularMarketChange"),
                        "change_percent": result.get("regularMarketChangePercent"),
                        "volume": result.get("regularMarketVolume"),
                        "market_cap": result.get("marketCap"),
                        "pe_ratio": result.get("priceEarnings"),
                        "dividend_yield": result.get("dividendYield"),
                        "high_52w": result.get("fiftyTwoWeekHigh"),
                        "low_52w": result.get("fiftyTwoWeekLow"),
                        "updated_at": datetime.now().isoformat(),
                        "source": "brapi",
                    }
        return None

    async def _get_yfinance_quote(self, ticker: str) -> dict | None:
        """Get quote using yfinance."""
        try:
            import yfinance as yf

            # Add .SA suffix for Brazilian stocks
            yf_ticker = ticker if "." in ticker else f"{ticker}.SA"
            stock = yf.Ticker(yf_ticker)
            info = stock.info

            if not info or "regularMarketPrice" not in info:
                return None

            return {
                "ticker": ticker,
                "name": info.get("longName", info.get("shortName", "")),
                "price": info.get("regularMarketPrice") or info.get("currentPrice"),
                "change": info.get("regularMarketChange"),
                "change_percent": info.get("regularMarketChangePercent"),
                "volume": info.get("regularMarketVolume") or info.get("volume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "high_52w": info.get("fiftyTwoWeekHigh"),
                "low_52w": info.get("fiftyTwoWeekLow"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "updated_at": datetime.now().isoformat(),
                "source": "yfinance",
            }
        except Exception as e:
            logger.error("YFinance error", ticker=ticker, error=str(e))
            return None

    async def get_stock_history(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> dict:
        """Get historical price data."""
        try:
            import yfinance as yf
            import pandas as pd

            yf_ticker = ticker if "." in ticker else f"{ticker.upper()}.SA"
            stock = yf.Ticker(yf_ticker)
            hist = stock.history(period=period, interval=interval)

            if hist.empty:
                return {"error": f"Nenhum dado historico encontrado para {ticker}"}

            # Convert to list of records
            records = []
            for date, row in hist.iterrows():
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(row["Open"], 2),
                    "high": round(row["High"], 2),
                    "low": round(row["Low"], 2),
                    "close": round(row["Close"], 2),
                    "volume": int(row["Volume"]),
                })

            return {
                "ticker": ticker.upper(),
                "period": period,
                "interval": interval,
                "data": records,
                "count": len(records),
            }
        except Exception as e:
            logger.error("History error", ticker=ticker, error=str(e))
            return {"error": f"Erro ao obter historico: {str(e)}"}

    async def get_market_indices(self) -> dict:
        """Get major market indices."""
        indices = [
            ("^BVSP", "IBOVESPA"),
            ("IFIX.SA", "IFIX"),
            ("^GSPC", "S&P 500"),
            ("USDBRL=X", "Dolar"),
            ("EURBRL=X", "Euro"),
            ("BTC-BRL", "Bitcoin"),
        ]

        results = []
        try:
            import yfinance as yf

            for symbol, name in indices:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="2d")

                    if not hist.empty:
                        current = hist["Close"].iloc[-1]
                        previous = hist["Close"].iloc[-2] if len(hist) > 1 else current
                        change = current - previous
                        change_pct = (change / previous * 100) if previous else 0

                        results.append({
                            "symbol": symbol,
                            "name": name,
                            "value": round(current, 2),
                            "change": round(change, 2),
                            "change_percent": round(change_pct, 2),
                        })
                except Exception as e:
                    logger.warning("Index fetch failed", symbol=symbol, error=str(e))
                    continue

        except ImportError:
            return {"error": "yfinance not installed"}

        return {
            "indices": results,
            "updated_at": datetime.now().isoformat(),
        }

    async def search_stocks(
        self,
        query: str,
        asset_type: str = "all",
    ) -> dict:
        """Search for stocks by name or ticker."""
        # Use BRAPI search if available
        if settings.BRAPI_TOKEN:
            try:
                url = "https://brapi.dev/api/available"
                params = {"token": settings.BRAPI_TOKEN}

                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params=params, timeout=10.0)
                    if response.status_code == 200:
                        data = response.json()
                        stocks = data.get("stocks", [])

                        # Filter by query
                        query_upper = query.upper()
                        results = [
                            s for s in stocks
                            if query_upper in s.upper()
                        ][:20]

                        return {
                            "query": query,
                            "results": [{"ticker": s} for s in results],
                            "count": len(results),
                        }
            except Exception as e:
                logger.warning("Search failed", error=str(e))

        return {"query": query, "results": [], "count": 0}

    async def calculate_technical_indicators(
        self,
        ticker: str,
        indicators: list[str] | None = None,
        period: str = "3mo",
    ) -> dict:
        """Calculate technical indicators."""
        if indicators is None:
            indicators = ["rsi", "macd", "sma_20", "sma_50", "bollinger"]

        try:
            import yfinance as yf
            import pandas as pd
            import numpy as np

            yf_ticker = ticker if "." in ticker else f"{ticker.upper()}.SA"
            stock = yf.Ticker(yf_ticker)
            hist = stock.history(period=period)

            if hist.empty:
                return {"error": f"Nenhum dado encontrado para {ticker}"}

            close = hist["Close"]
            results = {"ticker": ticker.upper(), "indicators": {}}

            # RSI
            if "rsi" in indicators:
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                results["indicators"]["rsi"] = {
                    "value": round(rsi.iloc[-1], 2),
                    "interpretation": (
                        "Sobrecomprado" if rsi.iloc[-1] > 70
                        else "Sobrevendido" if rsi.iloc[-1] < 30
                        else "Neutro"
                    ),
                }

            # MACD
            if "macd" in indicators:
                exp1 = close.ewm(span=12, adjust=False).mean()
                exp2 = close.ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                histogram = macd - signal
                results["indicators"]["macd"] = {
                    "macd": round(macd.iloc[-1], 4),
                    "signal": round(signal.iloc[-1], 4),
                    "histogram": round(histogram.iloc[-1], 4),
                    "interpretation": "Alta" if histogram.iloc[-1] > 0 else "Baixa",
                }

            # SMA 20
            if "sma_20" in indicators:
                sma20 = close.rolling(window=20).mean()
                results["indicators"]["sma_20"] = {
                    "value": round(sma20.iloc[-1], 2),
                    "price_vs_sma": "Acima" if close.iloc[-1] > sma20.iloc[-1] else "Abaixo",
                }

            # SMA 50
            if "sma_50" in indicators:
                sma50 = close.rolling(window=50).mean()
                results["indicators"]["sma_50"] = {
                    "value": round(sma50.iloc[-1], 2),
                    "price_vs_sma": "Acima" if close.iloc[-1] > sma50.iloc[-1] else "Abaixo",
                }

            # Bollinger Bands
            if "bollinger" in indicators:
                sma = close.rolling(window=20).mean()
                std = close.rolling(window=20).std()
                upper = sma + (std * 2)
                lower = sma - (std * 2)
                results["indicators"]["bollinger"] = {
                    "upper": round(upper.iloc[-1], 2),
                    "middle": round(sma.iloc[-1], 2),
                    "lower": round(lower.iloc[-1], 2),
                    "price_position": (
                        "Acima da banda superior" if close.iloc[-1] > upper.iloc[-1]
                        else "Abaixo da banda inferior" if close.iloc[-1] < lower.iloc[-1]
                        else "Dentro das bandas"
                    ),
                }

            results["current_price"] = round(close.iloc[-1], 2)
            results["period"] = period

            return results

        except ImportError:
            return {"error": "Dependencias nao instaladas (yfinance, pandas, numpy)"}
        except Exception as e:
            logger.error("Technical analysis error", ticker=ticker, error=str(e))
            return {"error": f"Erro ao calcular indicadores: {str(e)}"}

    async def get_company_fundamentals(self, ticker: str) -> dict:
        """Get company fundamental data."""
        try:
            import yfinance as yf

            yf_ticker = ticker if "." in ticker else f"{ticker.upper()}.SA"
            stock = yf.Ticker(yf_ticker)
            info = stock.info

            if not info:
                return {"error": f"Dados nao encontrados para {ticker}"}

            return {
                "ticker": ticker.upper(),
                "name": info.get("longName", info.get("shortName", "")),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "fundamentals": {
                    "market_cap": info.get("marketCap"),
                    "enterprise_value": info.get("enterpriseValue"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "price_to_book": info.get("priceToBook"),
                    "price_to_sales": info.get("priceToSalesTrailing12Months"),
                    "ev_to_ebitda": info.get("enterpriseToEbitda"),
                    "ev_to_revenue": info.get("enterpriseToRevenue"),
                },
                "profitability": {
                    "profit_margin": info.get("profitMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "gross_margin": info.get("grossMargins"),
                    "roe": info.get("returnOnEquity"),
                    "roa": info.get("returnOnAssets"),
                },
                "dividends": {
                    "dividend_yield": info.get("dividendYield"),
                    "dividend_rate": info.get("dividendRate"),
                    "payout_ratio": info.get("payoutRatio"),
                    "ex_dividend_date": info.get("exDividendDate"),
                },
                "financials": {
                    "revenue": info.get("totalRevenue"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "earnings": info.get("netIncomeToCommon"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "free_cash_flow": info.get("freeCashflow"),
                    "operating_cash_flow": info.get("operatingCashflow"),
                    "total_debt": info.get("totalDebt"),
                    "total_cash": info.get("totalCash"),
                    "debt_to_equity": info.get("debtToEquity"),
                },
                "trading": {
                    "beta": info.get("beta"),
                    "avg_volume": info.get("averageVolume"),
                    "avg_volume_10d": info.get("averageVolume10days"),
                    "shares_outstanding": info.get("sharesOutstanding"),
                    "float_shares": info.get("floatShares"),
                },
            }
        except Exception as e:
            logger.error("Fundamentals error", ticker=ticker, error=str(e))
            return {"error": f"Erro ao obter dados fundamentalistas: {str(e)}"}

    async def get_dividends(self, ticker: str, years: int = 5) -> dict:
        """Get dividend history."""
        try:
            import yfinance as yf
            from datetime import datetime, timedelta

            yf_ticker = ticker if "." in ticker else f"{ticker.upper()}.SA"
            stock = yf.Ticker(yf_ticker)

            # Get dividends
            dividends = stock.dividends
            if dividends.empty:
                return {"ticker": ticker.upper(), "dividends": [], "message": "Nenhum dividendo encontrado"}

            # Filter by years
            cutoff = datetime.now() - timedelta(days=years * 365)
            dividends = dividends[dividends.index >= cutoff]

            records = []
            for date, value in dividends.items():
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "value": round(float(value), 4),
                })

            # Calculate stats
            total = sum(d["value"] for d in records)
            avg_annual = total / years if years > 0 else 0

            return {
                "ticker": ticker.upper(),
                "dividends": records,
                "total": round(total, 4),
                "count": len(records),
                "avg_annual": round(avg_annual, 4),
                "years": years,
            }
        except Exception as e:
            logger.error("Dividends error", ticker=ticker, error=str(e))
            return {"error": f"Erro ao obter dividendos: {str(e)}"}

    async def get_market_news(
        self,
        ticker: str | None = None,
        limit: int = 10,
    ) -> dict:
        """Get market news."""
        try:
            import yfinance as yf

            if ticker:
                yf_ticker = ticker if "." in ticker else f"{ticker.upper()}.SA"
                stock = yf.Ticker(yf_ticker)
                news = stock.news
            else:
                # Get general market news from IBOVESPA
                stock = yf.Ticker("^BVSP")
                news = stock.news

            if not news:
                return {"news": [], "message": "Nenhuma noticia encontrada"}

            articles = []
            for item in news[:limit]:
                articles.append({
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "link": item.get("link"),
                    "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat(),
                    "type": item.get("type"),
                })

            return {
                "ticker": ticker,
                "news": articles,
                "count": len(articles),
            }
        except Exception as e:
            logger.error("News error", error=str(e))
            return {"error": f"Erro ao obter noticias: {str(e)}"}

    async def get_selic_rate(self) -> dict:
        """Get SELIC rate."""
        try:
            # Using BCB API
            url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/10?formato=json"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    latest = data[-1] if data else None

                    return {
                        "rate": float(latest["valor"]) if latest else None,
                        "date": latest.get("data") if latest else None,
                        "history": [
                            {"date": d["data"], "rate": float(d["valor"])}
                            for d in data
                        ],
                        "source": "Banco Central do Brasil",
                    }
        except Exception as e:
            logger.error("SELIC error", error=str(e))

        return {"error": "Erro ao obter taxa SELIC"}

    async def get_inflation_data(self, index: str = "ipca") -> dict:
        """Get inflation data from BCB."""
        # BCB series codes
        series_map = {
            "ipca": 433,    # IPCA - Variacao mensal
            "igpm": 189,    # IGP-M - Variacao mensal
            "inpc": 188,    # INPC - Variacao mensal
        }

        series_id = series_map.get(index.lower(), 433)

        try:
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados/ultimos/12?formato=json"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    latest = data[-1] if data else None

                    # Calculate 12-month accumulated
                    accumulated = sum(float(d["valor"]) for d in data)

                    return {
                        "index": index.upper(),
                        "latest": {
                            "date": latest.get("data") if latest else None,
                            "value": float(latest["valor"]) if latest else None,
                        },
                        "accumulated_12m": round(accumulated, 2),
                        "history": [
                            {"date": d["data"], "value": float(d["valor"])}
                            for d in data
                        ],
                        "source": "Banco Central do Brasil",
                    }
        except Exception as e:
            logger.error("Inflation error", index=index, error=str(e))

        return {"error": f"Erro ao obter dados de {index.upper()}"}

    async def compare_stocks(self, tickers: list[str]) -> dict:
        """Compare multiple stocks."""
        if len(tickers) > 5:
            tickers = tickers[:5]

        results = []
        for ticker in tickers:
            data = await self.get_company_fundamentals(ticker)
            if "error" not in data:
                results.append({
                    "ticker": ticker.upper(),
                    "name": data.get("name"),
                    "pe_ratio": data.get("fundamentals", {}).get("pe_ratio"),
                    "price_to_book": data.get("fundamentals", {}).get("price_to_book"),
                    "dividend_yield": data.get("dividends", {}).get("dividend_yield"),
                    "roe": data.get("profitability", {}).get("roe"),
                    "profit_margin": data.get("profitability", {}).get("profit_margin"),
                    "debt_to_equity": data.get("financials", {}).get("debt_to_equity"),
                    "market_cap": data.get("fundamentals", {}).get("market_cap"),
                })

        return {
            "comparison": results,
            "count": len(results),
        }

    def get_available_tools(self) -> list[dict]:
        """Get all available tools in OpenAI format."""
        return tool_registry.to_openai_format()

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        return await tool_registry.execute(tool_name, **kwargs)


# Singleton instance
mcp_client = MCPClient()
