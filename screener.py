"""
MomentumForge Stock Screener Backend
Scans NASDAQ/NYSE for momentum signals using free data sources.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Callable
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StockSignal:
    ticker: str
    name: str
    sector: str
    price: float
    change_percent: float
    market_cap_b: float
    
    # Technical signals
    above_ma20: bool
    above_ma50: bool
    above_ma200: bool
    volume_spike_pct: float
    
    # Fundamental signals
    earnings_surprise_pct: Optional[float]
    revenue_growth_yoy: Optional[float]
    revenue_accelerating: bool
    
    # Options signals
    options_volume_ratio: Optional[float]  # call/put ratio
    unusual_options_activity: bool
    options_flow_description: str
    
    # Sentiment
    analyst_rating: Optional[str]
    analyst_count: int

    # Computed
    signals: List[str] = field(default_factory=list)
    momentum_score: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert numpy types to Python native types for JSON serialization
        for key, value in d.items():
            if isinstance(value, (np.bool_, np.integer)):
                d[key] = int(value) if isinstance(value, np.integer) else bool(value)
            elif isinstance(value, np.floating):
                d[key] = float(value)
        return d


class MomentumScreener:
    """
    Screens stocks for momentum signals using:
    - Yahoo Finance for price, volume, technicals, options
    - Financial Modeling Prep for earnings/revenue (free tier)
    """
    
    FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    def __init__(self, fmp_api_key: Optional[str] = None):
        """
        Initialize screener.

        Args:
            fmp_api_key: Financial Modeling Prep API key (required for stock universe and fundamentals)
        """
        self.fmp_api_key = fmp_api_key
        self._ticker_cache = {}
        self._universe_cache = None
    
    def get_universe(self, min_market_cap_b: float = 5.0) -> List[str]:
        """
        Get full list of US-traded stocks from NYSE and NASDAQ.
        Fetches from FMP's stock list endpoint, with Wikipedia fallback.
        """
        # Return cached universe if available
        if self._universe_cache:
            return self._universe_cache

        universe = []

        # Method 1: Try FMP's full stock list (available on free tier)
        if self.fmp_api_key:
            try:
                stock_list_url = f"{self.FMP_BASE_URL}/stock/list?apikey={self.fmp_api_key}"
                response = requests.get(stock_list_url, timeout=60)
                logger.info(f"FMP stock list API response status: {response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        for stock in data:
                            symbol = stock.get('symbol', '')
                            exchange = stock.get('exchangeShortName', '')
                            stock_type = stock.get('type', '')

                            # Filter: US exchanges, stocks only, no special symbols
                            if (exchange in ['NYSE', 'NASDAQ', 'AMEX'] and
                                stock_type == 'stock' and
                                symbol and
                                '.' not in symbol and  # No class shares like BRK.A
                                '-' not in symbol and  # No warrants like SPAC-WT
                                '^' not in symbol and  # No indices
                                len(symbol) <= 5):     # Normal ticker length
                                universe.append(symbol)

                        logger.info(f"Fetched {len(universe)} US stocks from FMP stock list")

            except Exception as e:
                logger.error(f"Error fetching stock list from FMP: {e}")

        # Method 2: Try Wikipedia for S&P indices (covers large/mid/small caps)
        if not universe:
            logger.info("Trying Wikipedia for stock universe...")
            universe = self._fetch_wikipedia_stocks()

        # Method 3: Fallback to curated list
        if not universe:
            logger.warning("Using fallback stock universe (200 stocks)")
            universe = self._get_fallback_universe()

        # Remove duplicates and sort
        universe = sorted(list(set(universe)))

        logger.info(f"Total universe: {len(universe)} stocks")
        self._universe_cache = universe
        return universe

    def _fetch_wikipedia_stocks(self) -> List[str]:
        """
        Fetch stock lists from Wikipedia (S&P 500, S&P 400, S&P 600).
        This gives ~1,500 stocks covering large, mid, and small caps.
        """
        universe = []

        wiki_sources = [
            ('S&P 500', 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'),
            ('S&P 400', 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'),
            ('S&P 600', 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'),
        ]

        for name, url in wiki_sources:
            try:
                response = requests.get(url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; MomentumScreener/1.0)'
                })

                if response.status_code == 200:
                    # Parse HTML table - ticker is in first column
                    # Using pandas to parse HTML tables
                    tables = pd.read_html(response.text)
                    if tables:
                        df = tables[0]
                        # Find the symbol column (usually 'Symbol' or 'Ticker')
                        symbol_col = None
                        for col in df.columns:
                            col_str = str(col).lower()
                            if 'symbol' in col_str or 'ticker' in col_str:
                                symbol_col = col
                                break

                        if symbol_col is None and len(df.columns) > 0:
                            symbol_col = df.columns[0]  # Assume first column

                        if symbol_col is not None:
                            symbols = df[symbol_col].astype(str).tolist()
                            for sym in symbols:
                                # Clean up symbol
                                sym = sym.strip().upper()
                                if sym and len(sym) <= 5 and sym.isalpha():
                                    universe.append(sym)

                            logger.info(f"Fetched {len(symbols)} stocks from Wikipedia {name}")

            except Exception as e:
                logger.warning(f"Error fetching {name} from Wikipedia: {e}")

        return universe

    def _get_fallback_universe(self) -> List[str]:
        """
        Fallback list of 200 quality stocks across market caps.
        """
        return [
            # Mega caps (30)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'XOM', 'JPM', 'WMT', 'MA', 'PG', 'HD', 'CVX', 'MRK', 'ABBV',
            'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'MCD', 'CSCO', 'ACN', 'ABT',
            # Large cap tech/growth (30)
            'ADBE', 'ORCL', 'NFLX', 'INTC', 'QCOM', 'INTU', 'IBM', 'NOW', 'AMAT', 'ADP',
            'CRM', 'AMD', 'ISRG', 'BKNG', 'MDLZ', 'ADI', 'REGN', 'VRTX', 'GILD', 'PANW',
            'PLTR', 'COIN', 'SNOW', 'NET', 'SHOP', 'SQ', 'UBER', 'ABNB', 'DASH', 'CRWD',
            # Large cap value/dividend (30)
            'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB', 'PNC',
            'T', 'VZ', 'TMUS', 'CMCSA', 'DIS', 'NFLX', 'CHTR', 'EA', 'TTWO', 'WBD',
            'UPS', 'FDX', 'CAT', 'DE', 'BA', 'RTX', 'LMT', 'GE', 'HON', 'MMM',
            # Mid caps - Tech (20)
            'DDOG', 'ZS', 'MDB', 'OKTA', 'HUBS', 'TWLO', 'U', 'BILL', 'DOCN', 'PATH',
            'GTLB', 'ESTC', 'CFLT', 'SUMO', 'NEWR', 'DT', 'FIVN', 'RNG', 'ZI', 'ASAN',
            # Mid caps - Healthcare (20)
            'DXCM', 'IDXX', 'IQV', 'ALGN', 'HOLX', 'TECH', 'MTD', 'WAT', 'BIO', 'PKI',
            'VEEV', 'CDNS', 'ANSS', 'PAYC', 'CPRT', 'ODFL', 'POOL', 'WST', 'MPWR', 'LULU',
            # Mid caps - Industrials/Consumer (20)
            'AXON', 'TRMB', 'GNRC', 'ZBRA', 'TER', 'ENTG', 'SWKS', 'FFIV', 'JBHT', 'CHRW',
            'EXPD', 'LSTR', 'SAIA', 'XPO', 'WERN', 'KNX', 'HUBG', 'SNDR', 'ARCB', 'MATX',
            # Small caps - Growth (25)
            'CELH', 'ONON', 'DUOL', 'RKLB', 'IONQ', 'AFRM', 'SOFI', 'UPST', 'HOOD', 'OPEN',
            'RIVN', 'LCID', 'FSR', 'CHPT', 'BLNK', 'EVGO', 'PTRA', 'NKLA', 'HYLN', 'GOEV',
            'DNA', 'BEAM', 'CRSP', 'EDIT', 'NTLA',
            # Small caps - Value (25)
            'TMHC', 'MTH', 'CCS', 'MHO', 'TPH', 'KBH', 'MDC', 'LGIH', 'CVCO', 'SKY',
            'DDS', 'M', 'JWN', 'KSS', 'BURL', 'ROST', 'TJX', 'GPS', 'ANF', 'AEO',
            'CASY', 'ULTA', 'FIVE', 'OLLI', 'DG',
        ]
    
    def get_stock_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive stock data from Yahoo Finance.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Basic validation
            market_cap = info.get('marketCap', 0)
            if not market_cap or market_cap < 5e9:  # < $5B
                return None
            
            # Get historical data for technicals
            hist = stock.history(period='1y')
            if hist.empty or len(hist) < 200:
                return None
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            # Calculate moving averages
            ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            
            # Volume analysis
            avg_volume_20d = hist['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            volume_spike_pct = (current_volume / avg_volume_20d * 100) if avg_volume_20d > 0 else 100
            
            # Price change
            change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
            
            return {
                'ticker': ticker,
                'name': info.get('shortName', info.get('longName', ticker)),
                'sector': info.get('sector', 'Unknown'),
                'price': current_price,
                'change_percent': change_pct,
                'market_cap_b': market_cap / 1e9,
                'above_ma20': current_price > ma20,
                'above_ma50': current_price > ma50,
                'above_ma200': current_price > ma200,
                'volume_spike_pct': volume_spike_pct,
                'avg_volume': avg_volume_20d,
                'current_volume': current_volume,
                # Analyst data from Yahoo
                'analyst_rating': info.get('recommendationKey', None),
                'analyst_count': info.get('numberOfAnalystOpinions', 0),
                'target_price': info.get('targetMeanPrice', None),
                # Earnings data from Yahoo
                'trailing_eps': info.get('trailingEps', None),
                'forward_eps': info.get('forwardEps', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'revenue_growth': info.get('revenueGrowth', None),
            }
            
        except Exception as e:
            logger.warning(f"Error fetching {ticker}: {e}")
            return None
    
    def get_options_data(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze options activity for unusual signals.
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get nearest expiration
            expirations = stock.options
            if not expirations:
                return {
                    'options_volume_ratio': None,
                    'unusual_options_activity': False,
                    'options_flow_description': 'No options data'
                }
            
            # Get options chain for nearest expiration
            nearest_exp = expirations[0]
            opt_chain = stock.option_chain(nearest_exp)
            
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            total_call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
            total_put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
            
            # Handle NaN values
            total_call_volume = 0 if pd.isna(total_call_volume) else total_call_volume
            total_put_volume = 0 if pd.isna(total_put_volume) else total_put_volume
            
            # Calculate put/call ratio (inverted for bullish signal)
            if total_put_volume > 0:
                call_put_ratio = total_call_volume / total_put_volume
            else:
                call_put_ratio = float('inf') if total_call_volume > 0 else 1.0
            
            # Determine unusual activity
            unusual = False
            description = 'Neutral'
            
            if call_put_ratio > 2.0:
                unusual = True
                description = 'Heavy call buying'
            elif call_put_ratio > 1.5:
                description = 'Bullish call flow'
            elif call_put_ratio < 0.5:
                description = 'Heavy put activity'
            elif call_put_ratio < 0.7:
                description = 'Bearish put flow'
            else:
                description = 'Balanced flow'
            
            # Check for unusual volume in OTM calls (bullish signal)
            if not calls.empty:
                otm_calls = calls[calls['inTheMoney'] == False]
                if not otm_calls.empty and 'volume' in otm_calls.columns:
                    otm_volume = otm_calls['volume'].sum()
                    if pd.notna(otm_volume) and otm_volume > total_call_volume * 0.6:
                        unusual = True
                        description = 'Unusual OTM call activity'
            
            return {
                'options_volume_ratio': round(call_put_ratio, 2) if call_put_ratio != float('inf') else 99.0,
                'unusual_options_activity': unusual,
                'options_flow_description': description
            }
            
        except Exception as e:
            logger.warning(f"Error fetching options for {ticker}: {e}")
            return {
                'options_volume_ratio': None,
                'unusual_options_activity': False,
                'options_flow_description': 'Error fetching options'
            }
    
    def get_earnings_data_fmp(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch earnings/revenue data from Financial Modeling Prep.
        """
        if not self.fmp_api_key:
            return {
                'earnings_surprise_pct': None,
                'revenue_growth_yoy': None,
                'revenue_accelerating': False
            }
        
        try:
            # Get earnings surprises
            url = f"{self.FMP_BASE_URL}/earnings-surprises/{ticker}?apikey={self.fmp_api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    latest = data[0]
                    actual = latest.get('actualEarningResult', 0)
                    estimated = latest.get('estimatedEarning', 0)
                    
                    if estimated and estimated != 0:
                        surprise_pct = ((actual - estimated) / abs(estimated)) * 100
                    else:
                        surprise_pct = None
                else:
                    surprise_pct = None
            else:
                surprise_pct = None
            
            # Get income statement for revenue growth
            url = f"{self.FMP_BASE_URL}/income-statement/{ticker}?period=quarter&limit=8&apikey={self.fmp_api_key}"
            response = requests.get(url, timeout=10)
            
            revenue_growth = None
            revenue_accelerating = False
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) >= 5:
                    # Current quarter vs same quarter last year
                    current_rev = data[0].get('revenue', 0)
                    yoy_rev = data[4].get('revenue', 0) if len(data) > 4 else 0
                    
                    if yoy_rev and yoy_rev > 0:
                        revenue_growth = ((current_rev - yoy_rev) / yoy_rev) * 100
                    
                    # Check if accelerating (this quarter's YoY > last quarter's YoY)
                    if len(data) >= 6:
                        prev_quarter_rev = data[1].get('revenue', 0)
                        prev_yoy_rev = data[5].get('revenue', 0)
                        
                        if prev_yoy_rev and prev_yoy_rev > 0:
                            prev_growth = ((prev_quarter_rev - prev_yoy_rev) / prev_yoy_rev) * 100
                            revenue_accelerating = revenue_growth is not None and revenue_growth > prev_growth
            
            return {
                'earnings_surprise_pct': round(surprise_pct, 1) if surprise_pct else None,
                'revenue_growth_yoy': round(revenue_growth, 1) if revenue_growth else None,
                'revenue_accelerating': revenue_accelerating
            }
            
        except Exception as e:
            logger.warning(f"Error fetching FMP data for {ticker}: {e}")
            return {
                'earnings_surprise_pct': None,
                'revenue_growth_yoy': None,
                'revenue_accelerating': False
            }
    
    def calculate_momentum_score(self, data: Dict[str, Any]) -> int:
        """
        Calculate composite momentum score (0-100).
        """
        score = 0
        
        # Technical signals (35 points max)
        if data.get('above_ma20'):
            score += 8
        if data.get('above_ma50'):
            score += 12
        if data.get('above_ma200'):
            score += 15
        
        # Volume (15 points max)
        volume_spike = data.get('volume_spike_pct', 100)
        if volume_spike >= 200:
            score += 15
        elif volume_spike >= 150:
            score += 10
        elif volume_spike >= 120:
            score += 5
        
        # Earnings surprise (12 points max)
        earnings_surprise = data.get('earnings_surprise_pct')
        if earnings_surprise is not None:
            if earnings_surprise >= 20:
                score += 12
            elif earnings_surprise >= 10:
                score += 8
            elif earnings_surprise >= 5:
                score += 4
        
        # Revenue growth (13 points max)
        revenue_growth = data.get('revenue_growth_yoy')
        if revenue_growth is not None:
            if revenue_growth >= 30:
                score += 8
            elif revenue_growth >= 15:
                score += 5
            elif revenue_growth >= 5:
                score += 3
        
        if data.get('revenue_accelerating'):
            score += 5
        
        # Options flow (15 points max)
        if data.get('unusual_options_activity'):
            score += 10

        options_ratio = data.get('options_volume_ratio')
        if options_ratio is not None and options_ratio > 1.5:
            score += 5

        return max(0, min(score, 100))
    
    def determine_signals(self, data: Dict[str, Any]) -> List[str]:
        """
        Determine which signals are active for a stock.
        """
        signals = []
        
        # Earnings signal
        earnings_surprise = data.get('earnings_surprise_pct')
        if earnings_surprise is not None and earnings_surprise >= 5:
            signals.append('earnings')
        
        # Revenue signal
        if data.get('revenue_growth_yoy') and data['revenue_growth_yoy'] >= 10:
            signals.append('revenue')
        
        # Volume signal
        if data.get('volume_spike_pct', 100) >= 150:
            signals.append('volume')
        
        # Options signal
        if data.get('unusual_options_activity') or (data.get('options_volume_ratio') and data['options_volume_ratio'] > 1.5):
            signals.append('options')

        return signals
    
    def analyze_stock(self, ticker: str) -> Optional[StockSignal]:
        """
        Full analysis of a single stock.

        Args:
            ticker: Stock ticker symbol
        """
        # Get base data
        data = self.get_stock_data(ticker)
        if not data:
            return None
        
        # Check technical filter: must be above all MAs
        if not (data['above_ma20'] and data['above_ma50'] and data['above_ma200']):
            return None
        
        # Get options data
        options_data = self.get_options_data(ticker)
        data.update(options_data)
        
        # Get earnings/revenue data (if API key available)
        if self.fmp_api_key:
            earnings_data = self.get_earnings_data_fmp(ticker)
            data.update(earnings_data)
        else:
            # Use Yahoo data as fallback
            data['earnings_surprise_pct'] = None
            rev_growth = data.get('revenue_growth')
            data['revenue_growth_yoy'] = round(rev_growth * 100, 1) if rev_growth else None
            data['revenue_accelerating'] = False

        # Calculate score and signals
        data['momentum_score'] = self.calculate_momentum_score(data)
        data['signals'] = self.determine_signals(data)

        # Must have at least one signal
        if not data['signals']:
            return None

        # Create signal object
        return StockSignal(
            ticker=data['ticker'],
            name=data['name'],
            sector=data['sector'],
            price=round(data['price'], 2),
            change_percent=round(data['change_percent'], 2),
            market_cap_b=round(data['market_cap_b'], 1),
            above_ma20=data['above_ma20'],
            above_ma50=data['above_ma50'],
            above_ma200=data['above_ma200'],
            volume_spike_pct=round(data['volume_spike_pct'], 0),
            earnings_surprise_pct=data.get('earnings_surprise_pct'),
            revenue_growth_yoy=data.get('revenue_growth_yoy'),
            revenue_accelerating=data.get('revenue_accelerating', False),
            options_volume_ratio=data.get('options_volume_ratio'),
            unusual_options_activity=data.get('unusual_options_activity', False),
            options_flow_description=data.get('options_flow_description', 'Unknown'),
            analyst_rating=data.get('analyst_rating'),
            analyst_count=data.get('analyst_count', 0),
            signals=data['signals'],
            momentum_score=data['momentum_score']
        )
    
    def run_scan(self, min_market_cap_b: float = 5.0, max_workers: int = 10,
                 progress_callback=None) -> List[StockSignal]:
        """
        Run full market scan.

        Args:
            min_market_cap_b: Minimum market cap in billions
            max_workers: Number of parallel threads
            progress_callback: Optional callback for progress updates

        Returns:
            List of StockSignal objects, sorted by momentum score
        """
        universe = self.get_universe(min_market_cap_b)
        total = len(universe)
        results = []
        completed = 0

        logger.info(f"Starting scan of {total} stocks...")

        if progress_callback:
            progress_callback(f"Scanning {total} stocks...", 0)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(self.analyze_stock, ticker): ticker
                               for ticker in universe}

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                completed += 1

                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logger.info(f"[{completed}/{total}] {ticker}: Score {result.momentum_score}, Signals: {result.signals}")
                    else:
                        logger.debug(f"[{completed}/{total}] {ticker}: Filtered out")
                except Exception as e:
                    logger.warning(f"[{completed}/{total}] {ticker}: Error - {e}")

                if progress_callback and completed % 10 == 0:
                    progress_callback(f"Analyzed {completed}/{total} stocks...", completed/total)

        # Sort by momentum score descending
        results.sort(key=lambda x: x.momentum_score, reverse=True)

        logger.info(f"Scan complete. Found {len(results)} stocks with momentum signals.")

        return results
    
    def to_json(self, results: List[StockSignal]) -> str:
        """Convert results to JSON string."""
        return json.dumps([r.to_dict() for r in results], indent=2)
    
    def to_dataframe(self, results: List[StockSignal]) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in results])


def main():
    """Run the screener and output results."""
    import argparse

    parser = argparse.ArgumentParser(description='MomentumForge Stock Screener')
    parser.add_argument('--fmp-key', type=str, help='Financial Modeling Prep API key')
    parser.add_argument('--min-cap', type=float, default=5.0, help='Minimum market cap in billions')
    parser.add_argument('--output', type=str, default='results.json', help='Output file path')
    parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    args = parser.parse_args()

    screener = MomentumScreener(fmp_api_key=args.fmp_key)

    print("=" * 60)
    print("MomentumForge Stock Screener")
    print("=" * 60)
    print(f"Min Market Cap: ${args.min_cap}B")
    print(f"Output: {args.output}")
    print("=" * 60)
    print()

    results = screener.run_scan(min_market_cap_b=args.min_cap)

    if args.format == 'json':
        with open(args.output, 'w') as f:
            f.write(screener.to_json(results))
    else:
        df = screener.to_dataframe(results)
        df.to_csv(args.output, index=False)

    print()
    print("=" * 60)
    print(f"TOP 10 MOMENTUM STOCKS")
    print("=" * 60)

    for i, stock in enumerate(results[:10], 1):
        print(f"\n{i}. {stock.ticker} - {stock.name}")
        print(f"   Price: ${stock.price:.2f} ({stock.change_percent:+.1f}%)")
        print(f"   Market Cap: ${stock.market_cap_b:.1f}B")
        print(f"   Momentum Score: {stock.momentum_score}/100")
        print(f"   Signals: {', '.join(stock.signals)}")
        print(f"   Volume: {stock.volume_spike_pct:.0f}% of avg")
        if stock.earnings_surprise_pct:
            print(f"   Earnings Surprise: {stock.earnings_surprise_pct:+.1f}%")
        if stock.revenue_growth_yoy:
            print(f"   Revenue Growth: {stock.revenue_growth_yoy:+.1f}% YoY {'(accelerating)' if stock.revenue_accelerating else ''}")
        print(f"   Options Flow: {stock.options_flow_description}")

    print()
    print(f"Full results saved to {args.output}")


if __name__ == '__main__':
    main()
