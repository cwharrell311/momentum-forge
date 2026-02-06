"""
Alpaca Markets API Integration
Real-time and historical market data, options chains.
Free with Alpaca brokerage account.

Docs: https://alpaca.markets/docs/api-references/market-data-api/
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger(__name__)


class AlpacaClient:
    """
    Alpaca Markets API client for market data.

    Provides:
    - Real-time and historical stock prices
    - Options chains and quotes
    - No rate limits on most endpoints
    """

    # API endpoints
    DATA_URL = "https://data.alpaca.markets"
    PAPER_URL = "https://paper-api.alpaca.markets"

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize Alpaca client.

        Args:
            api_key: Alpaca API key (or set ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (or set ALPACA_SECRET_KEY env var)
        """
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY')
        self.secret_key = secret_key or os.environ.get('ALPACA_SECRET_KEY')

        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API keys not configured")

        self.headers = {
            "APCA-API-KEY-ID": self.api_key or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
        }

    def is_configured(self) -> bool:
        """Check if API keys are configured."""
        return bool(self.api_key and self.secret_key)

    def get_latest_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get latest quote for a stock.

        Returns:
            Dict with 'bid', 'ask', 'last', 'volume' etc.
        """
        if not self.is_configured():
            return None

        try:
            url = f"{self.DATA_URL}/v2/stocks/{ticker}/quotes/latest"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                quote = data.get('quote', {})
                return {
                    'bid': quote.get('bp'),
                    'ask': quote.get('ap'),
                    'bid_size': quote.get('bs'),
                    'ask_size': quote.get('as'),
                    'timestamp': quote.get('t'),
                }
            else:
                logger.warning(f"Alpaca quote error for {ticker}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error fetching Alpaca quote for {ticker}: {e}")
            return None

    def get_latest_trade(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get latest trade for a stock.

        Returns:
            Dict with 'price', 'size', 'timestamp'
        """
        if not self.is_configured():
            return None

        try:
            url = f"{self.DATA_URL}/v2/stocks/{ticker}/trades/latest"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                trade = data.get('trade', {})
                return {
                    'price': trade.get('p'),
                    'size': trade.get('s'),
                    'timestamp': trade.get('t'),
                }
            else:
                logger.warning(f"Alpaca trade error for {ticker}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error fetching Alpaca trade for {ticker}: {e}")
            return None

    def get_bars(self, ticker: str, timeframe: str = "1Day",
                 limit: int = 200) -> Optional[List[Dict[str, Any]]]:
        """
        Get historical bars (OHLCV data).

        Args:
            ticker: Stock symbol
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            limit: Number of bars to return (max 10000)

        Returns:
            List of bars with 'open', 'high', 'low', 'close', 'volume', 'timestamp'
        """
        if not self.is_configured():
            return None

        try:
            # Calculate start date (go back enough to get limit bars)
            if timeframe == "1Day":
                start = datetime.now() - timedelta(days=limit + 50)  # Extra for weekends
            else:
                start = datetime.now() - timedelta(days=30)

            url = f"{self.DATA_URL}/v2/stocks/{ticker}/bars"
            params = {
                "timeframe": timeframe,
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": limit,
                "adjustment": "split",  # Adjust for splits
            }

            response = requests.get(url, headers=self.headers, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                bars = data.get('bars', [])

                return [
                    {
                        'timestamp': bar.get('t'),
                        'open': bar.get('o'),
                        'high': bar.get('h'),
                        'low': bar.get('l'),
                        'close': bar.get('c'),
                        'volume': bar.get('v'),
                        'vwap': bar.get('vw'),
                    }
                    for bar in bars
                ]
            else:
                logger.warning(f"Alpaca bars error for {ticker}: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error fetching Alpaca bars for {ticker}: {e}")
            return None

    def get_snapshot(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get snapshot of current market data (quote + trade + daily bar).
        Most efficient single call for current price data.

        Returns:
            Dict with 'price', 'change', 'change_percent', 'volume', etc.
        """
        if not self.is_configured():
            return None

        try:
            url = f"{self.DATA_URL}/v2/stocks/{ticker}/snapshot"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()

                latest_trade = data.get('latestTrade', {})
                daily_bar = data.get('dailyBar', {})
                prev_daily_bar = data.get('prevDailyBar', {})
                minute_bar = data.get('minuteBar', {})

                current_price = latest_trade.get('p') or daily_bar.get('c')
                prev_close = prev_daily_bar.get('c')

                change = None
                change_percent = None
                if current_price and prev_close:
                    change = current_price - prev_close
                    change_percent = (change / prev_close) * 100

                return {
                    'price': current_price,
                    'prev_close': prev_close,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': daily_bar.get('v'),
                    'vwap': daily_bar.get('vw'),
                    'open': daily_bar.get('o'),
                    'high': daily_bar.get('h'),
                    'low': daily_bar.get('l'),
                }
            else:
                logger.debug(f"Alpaca snapshot error for {ticker}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error fetching Alpaca snapshot for {ticker}: {e}")
            return None

    def get_multi_snapshots(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get snapshots for multiple stocks in one call.
        Much more efficient than individual calls.

        Args:
            tickers: List of stock symbols (max 100 per call)

        Returns:
            Dict mapping ticker -> snapshot data
        """
        if not self.is_configured():
            return {}

        results = {}

        # Alpaca limits to ~100 symbols per request
        batch_size = 100
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]

            try:
                url = f"{self.DATA_URL}/v2/stocks/snapshots"
                params = {"symbols": ",".join(batch)}
                response = requests.get(url, headers=self.headers, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    for ticker, snapshot in data.items():
                        latest_trade = snapshot.get('latestTrade', {})
                        daily_bar = snapshot.get('dailyBar', {})
                        prev_daily_bar = snapshot.get('prevDailyBar', {})

                        current_price = latest_trade.get('p') or daily_bar.get('c')
                        prev_close = prev_daily_bar.get('c')

                        change = None
                        change_percent = None
                        if current_price and prev_close:
                            change = current_price - prev_close
                            change_percent = (change / prev_close) * 100

                        results[ticker] = {
                            'price': current_price,
                            'prev_close': prev_close,
                            'change': change,
                            'change_percent': change_percent,
                            'volume': daily_bar.get('v'),
                            'vwap': daily_bar.get('vw'),
                            'open': daily_bar.get('o'),
                            'high': daily_bar.get('h'),
                            'low': daily_bar.get('l'),
                        }
                else:
                    logger.warning(f"Alpaca multi-snapshot error: {response.status_code}")

            except Exception as e:
                logger.error(f"Error fetching Alpaca snapshots: {e}")

        return results

    def get_options_chain(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get options chain for a stock.

        Returns:
            Dict with 'calls' and 'puts' lists
        """
        if not self.is_configured():
            return None

        try:
            # Get available expirations
            url = f"{self.DATA_URL}/v1beta1/options/contracts/{ticker}"
            params = {"limit": 1000}
            response = requests.get(url, headers=self.headers, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                contracts = data.get('option_contracts', [])

                calls = []
                puts = []

                for contract in contracts:
                    option_data = {
                        'symbol': contract.get('symbol'),
                        'strike': contract.get('strike_price'),
                        'expiration': contract.get('expiration_date'),
                        'type': contract.get('type'),  # 'call' or 'put'
                    }

                    if contract.get('type') == 'call':
                        calls.append(option_data)
                    else:
                        puts.append(option_data)

                return {
                    'calls': calls,
                    'puts': puts,
                    'total_contracts': len(contracts),
                }
            else:
                logger.debug(f"Alpaca options error for {ticker}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error fetching Alpaca options for {ticker}: {e}")
            return None

    def get_options_quotes(self, option_symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get quotes for specific option contracts.

        Args:
            option_symbols: List of option symbols (e.g., ['AAPL230120C00150000'])

        Returns:
            Dict mapping option symbol -> quote data
        """
        if not self.is_configured():
            return {}

        results = {}

        try:
            url = f"{self.DATA_URL}/v1beta1/options/quotes/latest"
            params = {"symbols": ",".join(option_symbols[:100])}  # Limit batch size
            response = requests.get(url, headers=self.headers, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                quotes = data.get('quotes', {})

                for symbol, quote in quotes.items():
                    results[symbol] = {
                        'bid': quote.get('bp'),
                        'ask': quote.get('ap'),
                        'bid_size': quote.get('bs'),
                        'ask_size': quote.get('as'),
                    }
            else:
                logger.warning(f"Alpaca options quotes error: {response.status_code}")

        except Exception as e:
            logger.error(f"Error fetching Alpaca options quotes: {e}")

        return results

    def calculate_technicals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Calculate technical indicators from historical bars.

        Returns:
            Dict with 'ma20', 'ma50', 'ma200', 'above_ma20', etc.
        """
        bars = self.get_bars(ticker, timeframe="1Day", limit=250)
        if not bars or len(bars) < 200:
            return None

        closes = [bar['close'] for bar in bars]
        volumes = [bar['volume'] for bar in bars]

        # Current values
        current_price = closes[-1]
        current_volume = volumes[-1]

        # Moving averages
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50
        ma200 = sum(closes[-200:]) / 200

        # Average volume (20-day)
        avg_volume_20d = sum(volumes[-20:]) / 20
        volume_spike_pct = (current_volume / avg_volume_20d * 100) if avg_volume_20d > 0 else 100

        return {
            'current_price': current_price,
            'ma20': ma20,
            'ma50': ma50,
            'ma200': ma200,
            'above_ma20': current_price > ma20,
            'above_ma50': current_price > ma50,
            'above_ma200': current_price > ma200,
            'volume_spike_pct': volume_spike_pct,
            'avg_volume': avg_volume_20d,
            'current_volume': current_volume,
        }


# Convenience function
def get_alpaca_client() -> AlpacaClient:
    """Get a configured Alpaca client."""
    return AlpacaClient()


if __name__ == "__main__":
    # Test the Alpaca integration
    logging.basicConfig(level=logging.INFO)

    client = AlpacaClient()

    if not client.is_configured():
        print("Alpaca API keys not configured!")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("Or pass them to AlpacaClient(api_key='...', secret_key='...')")
        exit(1)

    print("\n=== Testing Alpaca API for AAPL ===")

    # Test snapshot
    snapshot = client.get_snapshot('AAPL')
    if snapshot:
        print(f"\nSnapshot:")
        print(f"  Price: ${snapshot['price']:.2f}")
        print(f"  Change: {snapshot['change_percent']:.2f}%")
        print(f"  Volume: {snapshot['volume']:,}")

    # Test technicals
    technicals = client.calculate_technicals('AAPL')
    if technicals:
        print(f"\nTechnicals:")
        print(f"  Above MA20: {technicals['above_ma20']}")
        print(f"  Above MA50: {technicals['above_ma50']}")
        print(f"  Above MA200: {technicals['above_ma200']}")
        print(f"  Volume Spike: {technicals['volume_spike_pct']:.0f}%")

    # Test options
    options = client.get_options_chain('AAPL')
    if options:
        print(f"\nOptions:")
        print(f"  Total contracts: {options['total_contracts']}")
        print(f"  Calls: {len(options['calls'])}")
        print(f"  Puts: {len(options['puts'])}")
