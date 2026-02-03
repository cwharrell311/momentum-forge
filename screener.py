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
class SocialSentiment:
    """Social sentiment data from web sources."""
    score: float  # -1 to 1 scale (bearish to bullish)
    volume: int  # Number of mentions found
    sources: List[str]  # Where mentions were found
    summary: str  # Brief description of sentiment
    trending: bool  # Is this stock getting unusual attention?
    key_topics: List[str]  # Main themes being discussed
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    
    # Social sentiment
    social_sentiment: Optional[SocialSentiment] = None
    social_score: float = 0.0
    social_volume: int = 0
    social_summary: str = ""
    social_trending: bool = False
    social_sources: List[str] = field(default_factory=list)
    
    # Computed
    signals: List[str] = field(default_factory=list)
    momentum_score: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Flatten social sentiment for frontend
        if self.social_sentiment:
            d['social_score'] = self.social_sentiment.score
            d['social_volume'] = self.social_sentiment.volume
            d['social_summary'] = self.social_sentiment.summary
            d['social_trending'] = self.social_sentiment.trending
            d['social_sources'] = self.social_sentiment.sources
            d['social_topics'] = self.social_sentiment.key_topics
        return d


class SocialSentimentScanner:
    """
    Scans social media and forums for stock sentiment.
    
    This class provides multiple methods for sentiment scanning:
    1. Web search integration (for use with Claude's web_search tool)
    2. Direct API calls to StockTwits (free)
    3. Reddit API (requires credentials)
    """
    
    STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"
    
    # Sentiment keywords for analysis
    BULLISH_KEYWORDS = [
        'buy', 'calls', 'moon', 'rocket', 'bullish', 'long', 'breakout',
        'accumulating', 'undervalued', 'growth', 'beat', 'crush', 'soar',
        'surge', 'rally', 'rip', 'pump', 'strong', 'upgrade', 'target raised',
        'beat estimates', 'outperform', 'buy the dip', 'loading', 'adding'
    ]
    
    BEARISH_KEYWORDS = [
        'sell', 'puts', 'dump', 'crash', 'bearish', 'short', 'breakdown',
        'overvalued', 'miss', 'tank', 'plunge', 'fall', 'drop', 'weak',
        'downgrade', 'target lowered', 'missed estimates', 'underperform',
        'avoid', 'red flag', 'warning', 'bubble', 'fraud', 'scam'
    ]
    
    def __init__(self, web_search_func: Optional[Callable] = None):
        """
        Initialize the scanner.
        
        Args:
            web_search_func: Optional function that performs web searches.
                             Should accept a query string and return search results.
                             When running with Claude, this would be the web_search tool.
        """
        self.web_search_func = web_search_func
    
    def analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using keyword matching.
        Returns score from -1 (bearish) to 1 (bullish).
        """
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        # Score from -1 to 1
        return (bullish_count - bearish_count) / total
    
    def extract_topics(self, text: str, ticker: str) -> List[str]:
        """Extract key topics being discussed."""
        topics = []
        text_lower = text.lower()
        
        topic_patterns = {
            'earnings': ['earnings', 'eps', 'revenue', 'quarterly', 'beat', 'miss'],
            'guidance': ['guidance', 'outlook', 'forecast', 'raised', 'lowered'],
            'product': ['product', 'launch', 'release', 'announcement', 'new'],
            'partnership': ['partnership', 'deal', 'contract', 'agreement'],
            'competition': ['competition', 'competitor', 'market share'],
            'valuation': ['valuation', 'overvalued', 'undervalued', 'pe ratio', 'price target'],
            'insider': ['insider', 'ceo', 'executive', 'bought', 'sold'],
            'short_squeeze': ['short squeeze', 'short interest', 'shorts', 'covering'],
            'technical': ['breakout', 'support', 'resistance', 'chart', 'pattern'],
            'macro': ['fed', 'interest rate', 'inflation', 'recession', 'economy']
        }
        
        for topic, keywords in topic_patterns.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
        
        return topics[:5]  # Return top 5 topics
    
    def get_stocktwits_sentiment(self, ticker: str) -> Optional[SocialSentiment]:
        """
        Fetch sentiment from StockTwits API (free, no auth required).
        """
        try:
            url = f"{self.STOCKTWITS_BASE}/streams/symbol/{ticker}.json"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            messages = data.get('messages', [])
            
            if not messages:
                return None
            
            # Analyze messages
            bullish = 0
            bearish = 0
            neutral = 0
            all_text = []
            
            for msg in messages[:30]:  # Analyze last 30 messages
                sentiment = msg.get('entities', {}).get('sentiment', {})
                if sentiment:
                    if sentiment.get('basic') == 'Bullish':
                        bullish += 1
                    elif sentiment.get('basic') == 'Bearish':
                        bearish += 1
                    else:
                        neutral += 1
                
                body = msg.get('body', '')
                all_text.append(body)
            
            total = bullish + bearish + neutral
            if total == 0:
                return None
            
            # Calculate sentiment score
            if bullish + bearish > 0:
                score = (bullish - bearish) / (bullish + bearish)
            else:
                # Fall back to text analysis
                combined_text = ' '.join(all_text)
                score = self.analyze_text_sentiment(combined_text)
            
            # Determine if trending (high message volume)
            symbol_data = data.get('symbol', {})
            watchlist_count = symbol_data.get('watchlist_count', 0)
            trending = len(messages) >= 20 or watchlist_count > 10000
            
            # Generate summary
            if score > 0.3:
                summary = f"Strongly bullish ({bullish} bullish vs {bearish} bearish)"
            elif score > 0:
                summary = f"Slightly bullish sentiment"
            elif score < -0.3:
                summary = f"Strongly bearish ({bearish} bearish vs {bullish} bullish)"
            elif score < 0:
                summary = f"Slightly bearish sentiment"
            else:
                summary = "Mixed/neutral sentiment"
            
            # Extract topics
            topics = self.extract_topics(' '.join(all_text), ticker)
            
            return SocialSentiment(
                score=round(score, 2),
                volume=len(messages),
                sources=['StockTwits'],
                summary=summary,
                trending=trending,
                key_topics=topics
            )
            
        except Exception as e:
            logger.warning(f"StockTwits error for {ticker}: {e}")
            return None
    
    def search_web_sentiment(self, ticker: str, company_name: str = "") -> Optional[SocialSentiment]:
        """
        Search the web for sentiment using the provided web search function.
        This is designed to work with Claude's web_search tool.
        
        When called from Claude, the web_search_func would be the actual web search.
        """
        if not self.web_search_func:
            return None
        
        try:
            # Search queries to run
            queries = [
                f"{ticker} stock reddit",
                f"{ticker} stock twitter sentiment",
                f"${ticker} stocktwits",
            ]
            
            all_results = []
            sources = set()
            
            for query in queries:
                try:
                    results = self.web_search_func(query)
                    if results:
                        all_results.extend(results)
                        # Track sources
                        for r in results:
                            url = r.get('url', '').lower()
                            if 'reddit' in url:
                                sources.add('Reddit')
                            elif 'twitter' in url or 'x.com' in url:
                                sources.add('X/Twitter')
                            elif 'stocktwits' in url:
                                sources.add('StockTwits')
                            elif 'yahoo' in url:
                                sources.add('Yahoo Finance')
                            elif 'seekingalpha' in url:
                                sources.add('Seeking Alpha')
                            else:
                                sources.add('Web')
                except Exception as e:
                    logger.warning(f"Search error for query '{query}': {e}")
                    continue
            
            if not all_results:
                return None
            
            # Combine all text for analysis
            combined_text = ' '.join([
                r.get('title', '') + ' ' + r.get('snippet', '')
                for r in all_results
            ])
            
            # Analyze sentiment
            score = self.analyze_text_sentiment(combined_text)
            topics = self.extract_topics(combined_text, ticker)
            
            # Check for trending indicators
            trending_keywords = ['trending', 'viral', 'surge', 'volume spike', 'everyone talking']
            trending = any(kw in combined_text.lower() for kw in trending_keywords)
            
            # Generate summary
            if score > 0.3:
                summary = "Strong bullish chatter across social media"
            elif score > 0:
                summary = "Mildly positive sentiment online"
            elif score < -0.3:
                summary = "Significant bearish discussion"
            elif score < 0:
                summary = "Slightly negative sentiment"
            else:
                summary = "Mixed sentiment across sources"
            
            if trending:
                summary += " (trending)"
            
            return SocialSentiment(
                score=round(score, 2),
                volume=len(all_results),
                sources=list(sources),
                summary=summary,
                trending=trending,
                key_topics=topics
            )
            
        except Exception as e:
            logger.warning(f"Web search sentiment error for {ticker}: {e}")
            return None
    
    def get_sentiment(self, ticker: str, company_name: str = "") -> Optional[SocialSentiment]:
        """
        Get social sentiment using all available methods.
        Tries StockTwits first (free API), then web search if available.
        """
        # Try StockTwits first (most reliable free source)
        sentiment = self.get_stocktwits_sentiment(ticker)
        
        # If we have web search, enhance with additional sources
        if self.web_search_func:
            web_sentiment = self.search_web_sentiment(ticker, company_name)
            
            if web_sentiment:
                if sentiment:
                    # Merge the two
                    combined_score = (sentiment.score + web_sentiment.score) / 2
                    combined_sources = list(set(sentiment.sources + web_sentiment.sources))
                    combined_topics = list(set(sentiment.key_topics + web_sentiment.key_topics))[:5]
                    combined_volume = sentiment.volume + web_sentiment.volume
                    
                    sentiment = SocialSentiment(
                        score=round(combined_score, 2),
                        volume=combined_volume,
                        sources=combined_sources,
                        summary=web_sentiment.summary if abs(web_sentiment.score) > abs(sentiment.score) else sentiment.summary,
                        trending=sentiment.trending or web_sentiment.trending,
                        key_topics=combined_topics
                    )
                else:
                    sentiment = web_sentiment
        
        return sentiment


class MomentumScreener:
    """
    Screens stocks for momentum signals using:
    - Yahoo Finance for price, volume, technicals, options
    - Financial Modeling Prep for earnings/revenue (free tier)
    """
    
    FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    def __init__(self, fmp_api_key: Optional[str] = None, web_search_func: Optional[Callable] = None):
        """
        Initialize screener.
        
        Args:
            fmp_api_key: Financial Modeling Prep API key (optional, for fundamentals)
            web_search_func: Optional web search function for social sentiment
        """
        self.fmp_api_key = fmp_api_key
        self.web_search_func = web_search_func
        self.social_scanner = SocialSentimentScanner(web_search_func=web_search_func)
        self._ticker_cache = {}
    
    def get_universe(self, min_market_cap_b: float = 1.0) -> List[str]:
        """
        Get list of NASDAQ/NYSE stocks above minimum market cap.
        Uses a curated list of liquid stocks for efficiency.
        """
        # In production, you'd fetch this from an API or database
        # For now, using a solid universe of liquid mid-to-large caps
        universe = [
            # Mega caps
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'XOM', 'JPM', 'WMT', 'MA', 'PG', 'HD', 'CVX', 'MRK', 'ABBV',
            'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'MCD', 'CSCO', 'ACN', 'ABT',
            'DHR', 'NEE', 'NKE', 'WFC', 'LIN', 'TXN', 'PM', 'UNP', 'CRM', 'AMD',
            
            # Large cap tech/growth
            'ADBE', 'ORCL', 'NFLX', 'INTC', 'QCOM', 'INTU', 'IBM', 'NOW', 'AMAT', 'ADP',
            'ISRG', 'BKNG', 'MDLZ', 'ADI', 'REGN', 'VRTX', 'GILD', 'PANW', 'LRCX', 'MU',
            'SNPS', 'CDNS', 'KLAC', 'MRVL', 'FTNT', 'CRWD', 'DDOG', 'ZS', 'TEAM', 'WDAY',
            
            # High growth / momentum names
            'SMCI', 'ARM', 'PLTR', 'COIN', 'SNOW', 'NET', 'SHOP', 'SQ', 'ROKU', 'UBER',
            'ABNB', 'DASH', 'RBLX', 'U', 'PATH', 'MDB', 'OKTA', 'TWLO', 'ZM', 'DOCU',
            'ANET', 'AXON', 'TOST', 'DUOL', 'APP', 'TTD', 'BILL', 'HUBS', 'VEEV', 'ANSS',
            'CPRT', 'ODFL', 'DECK', 'POOL', 'MPWR', 'ALGN', 'IDXX', 'CTAS', 'PAYX', 'FAST',
            
            # Financials
            'GS', 'MS', 'BLK', 'SCHW', 'C', 'AXP', 'PNC', 'USB', 'TFC', 'COF',
            
            # Healthcare
            'AMGN', 'BMY', 'PFE', 'MRNA', 'BIIB', 'ILMN', 'DXCM', 'EW', 'ZTS', 'SYK',
            
            # Consumer
            'SBUX', 'TGT', 'LOW', 'ROST', 'DG', 'DLTR', 'ORLY', 'AZO', 'ULTA', 'LULU',
            'CMG', 'YUM', 'DPZ', 'WING', 'CAVA', 'CELH', 'MNST',
            
            # Industrial
            'CAT', 'DE', 'HON', 'GE', 'BA', 'RTX', 'LMT', 'NOC', 'GD', 'MMM',
            
            # Energy
            'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY', 'DVN', 'HES',
        ]
        
        return universe
    
    def get_stock_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive stock data from Yahoo Finance.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Basic validation
            market_cap = info.get('marketCap', 0)
            if not market_cap or market_cap < 1e9:  # < $1B
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
        
        # Options flow (12 points max)
        if data.get('unusual_options_activity'):
            score += 8
        
        options_ratio = data.get('options_volume_ratio')
        if options_ratio is not None and options_ratio > 1.5:
            score += 4
        
        # Social sentiment (13 points max) - NEW
        social_sentiment = data.get('social_sentiment')
        if social_sentiment:
            # Bullish sentiment adds points, bearish subtracts
            sentiment_score = social_sentiment.score if isinstance(social_sentiment, SocialSentiment) else social_sentiment.get('score', 0)
            
            if sentiment_score > 0.5:
                score += 8
            elif sentiment_score > 0.2:
                score += 5
            elif sentiment_score > 0:
                score += 2
            elif sentiment_score < -0.3:
                score -= 3  # Penalty for bearish sentiment
            
            # Trending bonus
            is_trending = social_sentiment.trending if isinstance(social_sentiment, SocialSentiment) else social_sentiment.get('trending', False)
            if is_trending and sentiment_score > 0:
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
        
        # Social signal - NEW
        social_sentiment = data.get('social_sentiment')
        if social_sentiment:
            sentiment_score = social_sentiment.score if isinstance(social_sentiment, SocialSentiment) else social_sentiment.get('score', 0)
            is_trending = social_sentiment.trending if isinstance(social_sentiment, SocialSentiment) else social_sentiment.get('trending', False)
            volume = social_sentiment.volume if isinstance(social_sentiment, SocialSentiment) else social_sentiment.get('volume', 0)
            
            # Social signal triggers on: bullish sentiment OR trending with decent volume
            if sentiment_score > 0.2 or (is_trending and volume >= 10):
                signals.append('social')
        
        return signals
    
    def analyze_stock(self, ticker: str, include_social: bool = True) -> Optional[StockSignal]:
        """
        Full analysis of a single stock.
        
        Args:
            ticker: Stock ticker symbol
            include_social: Whether to scan social sentiment (slower but more complete)
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
        
        # Get social sentiment
        if include_social:
            social_sentiment = self.social_scanner.get_sentiment(ticker, data.get('name', ''))
            data['social_sentiment'] = social_sentiment
        else:
            data['social_sentiment'] = None
        
        # Calculate score and signals
        data['momentum_score'] = self.calculate_momentum_score(data)
        data['signals'] = self.determine_signals(data)
        
        # Must have at least one signal
        if not data['signals']:
            return None
        
        # Extract social data for the dataclass
        social = data.get('social_sentiment')
        
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
            social_sentiment=social,
            social_score=social.score if social else 0.0,
            social_volume=social.volume if social else 0,
            social_summary=social.summary if social else "",
            social_trending=social.trending if social else False,
            social_sources=social.sources if social else [],
            signals=data['signals'],
            momentum_score=data['momentum_score']
        )
    
    def run_scan(self, min_market_cap_b: float = 1.0, max_workers: int = 10, 
                 include_social: bool = True, progress_callback=None) -> List[StockSignal]:
        """
        Run full market scan.
        
        Args:
            min_market_cap_b: Minimum market cap in billions
            max_workers: Number of parallel threads
            include_social: Whether to scan social sentiment
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of StockSignal objects, sorted by momentum score
        """
        universe = self.get_universe(min_market_cap_b)
        total = len(universe)
        results = []
        completed = 0
        
        logger.info(f"Starting scan of {total} stocks (social sentiment: {include_social})...")
        
        if progress_callback:
            progress_callback(f"Scanning {total} stocks...", 0)
        
        # Use fewer workers for social scanning to avoid rate limits
        workers = max_workers if not include_social else min(max_workers, 5)
        
        def analyze_with_social(ticker):
            return self.analyze_stock(ticker, include_social=include_social)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_ticker = {executor.submit(analyze_with_social, ticker): ticker 
                               for ticker in universe}
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        social_info = f", Social: {result.social_summary}" if result.social_sentiment else ""
                        logger.info(f"[{completed}/{total}] {ticker}: Score {result.momentum_score}, Signals: {result.signals}{social_info}")
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
    parser.add_argument('--min-cap', type=float, default=1.0, help='Minimum market cap in billions')
    parser.add_argument('--output', type=str, default='results.json', help='Output file path')
    parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    parser.add_argument('--no-social', action='store_true', help='Skip social sentiment scanning')
    args = parser.parse_args()
    
    screener = MomentumScreener(fmp_api_key=args.fmp_key)
    
    print("=" * 60)
    print("MomentumForge Stock Screener")
    print("=" * 60)
    print(f"Min Market Cap: ${args.min_cap}B")
    print(f"Social Sentiment: {'Disabled' if args.no_social else 'Enabled'}")
    print(f"Output: {args.output}")
    print("=" * 60)
    print()
    
    results = screener.run_scan(
        min_market_cap_b=args.min_cap,
        include_social=not args.no_social
    )
    
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
        if stock.social_sentiment:
            print(f"   Social: {stock.social_summary} (score: {stock.social_score:+.2f})")
            if stock.social_sentiment.key_topics:
                print(f"   Topics: {', '.join(stock.social_sentiment.key_topics)}")
    
    print()
    print(f"Full results saved to {args.output}")


if __name__ == '__main__':
    main()
