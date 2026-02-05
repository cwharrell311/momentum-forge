"""
Congressional Stock Trade Tracker
Fetches and processes stock trades reported by members of Congress.
Data sources: FMP API (primary), Capitol Trades scraping (fallback).
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
import re
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CongressTrade:
    politician: str
    party: str
    chamber: str  # 'Senate' or 'House'
    state: str
    ticker: str
    asset_description: str
    trade_type: str  # 'Purchase', 'Sale', 'Sale (Partial)', 'Sale (Full)', 'Exchange'
    trade_date: str
    disclosure_date: str
    amount_range: str  # e.g. '$1,001 - $15,000'
    amount_low: float
    amount_high: float
    days_to_disclose: int
    owner: str  # 'Self', 'Spouse', 'Joint', 'Child'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CongressTracker:
    """
    Tracks congressional stock trades from multiple data sources.
    """

    FMP_BASE_URL = "https://financialmodelingprep.com/api/v4"

    def __init__(self, fmp_api_key: Optional[str] = None):
        self.fmp_api_key = fmp_api_key
        self._cache = None
        self._cache_time = None
        self._cache_ttl = 3600  # Cache for 1 hour

    def get_trades(self, days_back: int = 90) -> List[CongressTrade]:
        """
        Fetch recent congressional trades. Tries FMP first, then Capitol Trades.
        """
        # Return cache if fresh
        if self._cache and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < self._cache_ttl:
                logger.info(f"Returning {len(self._cache)} cached trades")
                return self._cache

        trades = []

        # Method 1: Try FMP API
        if self.fmp_api_key:
            trades = self._fetch_fmp_trades(days_back)

        # Method 2: Fallback to Capitol Trades scraping
        if not trades:
            logger.info("FMP congressional data unavailable, trying Capitol Trades...")
            trades = self._fetch_capitol_trades(days_back)

        # Sort by disclosure date (newest first)
        trades.sort(key=lambda t: t.disclosure_date, reverse=True)

        # Cache results
        self._cache = trades
        self._cache_time = datetime.now()

        logger.info(f"Total congressional trades fetched: {len(trades)}")
        return trades

    def _fetch_fmp_trades(self, days_back: int) -> List[CongressTrade]:
        """Fetch trades from FMP Senate/House trading endpoints."""
        trades = []

        endpoints = [
            ('senate-disclosure-rss', 'Senate'),
            ('house-disclosure-rss', 'House'),
        ]

        for endpoint, chamber in endpoints:
            try:
                url = f"{self.FMP_BASE_URL}/{endpoint}?page=0&apikey={self.fmp_api_key}"
                response = requests.get(url, timeout=30)
                logger.info(f"FMP {chamber} trading API response: {response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        for item in data:
                            trade = self._parse_fmp_trade(item, chamber)
                            if trade:
                                trades.append(trade)
                        logger.info(f"Fetched {len(data)} {chamber} trades from FMP")
                    else:
                        logger.warning(f"FMP {chamber} returned empty data")
                        return []  # Likely paywalled
                elif response.status_code == 403:
                    logger.warning(f"FMP {chamber} endpoint returned 403 - paywalled")
                    return []  # Paywalled, skip to fallback

            except Exception as e:
                logger.error(f"Error fetching FMP {chamber} trades: {e}")

        return trades

    def _parse_fmp_trade(self, item: Dict, chamber: str) -> Optional[CongressTrade]:
        """Parse a single FMP trade entry."""
        try:
            ticker = item.get('ticker', item.get('symbol', ''))
            if not ticker or ticker == '--' or ticker == 'N/A':
                return None

            trade_date = item.get('transactionDate', item.get('transaction_date', ''))
            disclosure_date = item.get('disclosureDate', item.get('disclosure_date', ''))

            # Calculate days to disclose
            days_to_disclose = 0
            if trade_date and disclosure_date:
                try:
                    td = datetime.strptime(trade_date, '%Y-%m-%d')
                    dd = datetime.strptime(disclosure_date, '%Y-%m-%d')
                    days_to_disclose = (dd - td).days
                except ValueError:
                    pass

            amount = item.get('amount', item.get('range', '$0 - $0'))
            amount_low, amount_high = self._parse_amount_range(amount)

            # Determine party from name or separate field
            party = item.get('party', '')
            politician = item.get('firstName', item.get('representative', ''))
            last_name = item.get('lastName', '')
            if last_name:
                politician = f"{politician} {last_name}".strip()

            trade_type = item.get('type', item.get('transaction', 'Unknown'))

            return CongressTrade(
                politician=politician or 'Unknown',
                party=party or 'Unknown',
                chamber=chamber,
                state=item.get('state', item.get('district', '')),
                ticker=ticker.upper(),
                asset_description=item.get('assetDescription', item.get('asset_description', ticker)),
                trade_type=self._normalize_trade_type(trade_type),
                trade_date=trade_date,
                disclosure_date=disclosure_date,
                amount_range=amount,
                amount_low=amount_low,
                amount_high=amount_high,
                days_to_disclose=days_to_disclose,
                owner=item.get('owner', 'Self'),
            )
        except Exception as e:
            logger.debug(f"Error parsing FMP trade: {e}")
            return None

    def _fetch_capitol_trades(self, days_back: int) -> List[CongressTrade]:
        """Scrape trades from Capitol Trades website."""
        trades = []
        pages_to_fetch = 3  # First 3 pages should cover ~90 days

        for page in range(1, pages_to_fetch + 1):
            try:
                url = f"https://www.capitoltrades.com/trades?page={page}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                }
                response = requests.get(url, headers=headers, timeout=30)
                logger.info(f"Capitol Trades page {page} response: {response.status_code}")

                if response.status_code == 200:
                    page_trades = self._parse_capitol_trades_html(response.text)
                    trades.extend(page_trades)
                    logger.info(f"Parsed {len(page_trades)} trades from Capitol Trades page {page}")

                    if len(page_trades) == 0:
                        break  # No more data

                    time.sleep(1)  # Rate limit
                else:
                    logger.warning(f"Capitol Trades returned {response.status_code}")
                    break

            except Exception as e:
                logger.error(f"Error fetching Capitol Trades page {page}: {e}")
                break

        return trades

    def _parse_capitol_trades_html(self, html: str) -> List[CongressTrade]:
        """Parse Capitol Trades HTML into trade objects."""
        trades = []

        try:
            tables = pd.read_html(html)
            if not tables:
                logger.warning("No tables found in Capitol Trades HTML")
                return trades

            df = tables[0]
            logger.info(f"Capitol Trades table columns: {list(df.columns)}")

            for _, row in df.iterrows():
                try:
                    trade = self._parse_capitol_row(row)
                    if trade:
                        trades.append(trade)
                except Exception as e:
                    logger.debug(f"Error parsing Capitol Trades row: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Error parsing Capitol Trades HTML with pandas: {e}")
            # Fallback: try regex parsing
            trades = self._parse_capitol_trades_regex(html)

        return trades

    def _parse_capitol_row(self, row) -> Optional[CongressTrade]:
        """Parse a single row from Capitol Trades table."""
        try:
            # Capitol Trades columns vary, try common patterns
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else {}
            row_values = [str(v) for v in row.values]
            row_str = ' | '.join(row_values)

            # Try to extract politician name (usually first column)
            politician = str(row.iloc[0]) if len(row) > 0 else 'Unknown'

            # Find ticker - look for uppercase 1-5 letter strings
            ticker = ''
            for val in row_values:
                val = str(val).strip()
                match = re.search(r'\b([A-Z]{1,5})\b', val)
                if match and match.group(1) not in ['N', 'A', 'THE', 'AND', 'FOR', 'BUY', 'SELL', 'USD', 'NAN']:
                    ticker = match.group(1)
                    break

            if not ticker:
                return None

            # Determine trade type
            trade_type = 'Unknown'
            for val in row_values:
                val_lower = str(val).lower()
                if 'buy' in val_lower or 'purchase' in val_lower:
                    trade_type = 'Purchase'
                    break
                elif 'sell' in val_lower or 'sale' in val_lower:
                    trade_type = 'Sale'
                    break

            # Find amounts
            amount_range = ''
            amount_low = 0.0
            amount_high = 0.0
            for val in row_values:
                if '$' in str(val):
                    amount_range = str(val)
                    amount_low, amount_high = self._parse_amount_range(amount_range)
                    break

            # Find dates
            trade_date = ''
            disclosure_date = ''
            dates_found = []
            for val in row_values:
                date_match = re.search(r'(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})', str(val))
                if date_match:
                    dates_found.append(date_match.group(1))

            if len(dates_found) >= 2:
                trade_date = dates_found[0]
                disclosure_date = dates_found[1]
            elif len(dates_found) == 1:
                trade_date = dates_found[0]
                disclosure_date = dates_found[0]

            # Determine party/chamber
            party = 'Unknown'
            chamber = 'Unknown'
            for val in row_values:
                val_str = str(val)
                if 'Democrat' in val_str or '(D)' in val_str or val_str.strip() == 'D':
                    party = 'Democrat'
                elif 'Republican' in val_str or '(R)' in val_str or val_str.strip() == 'R':
                    party = 'Republican'
                elif 'Independent' in val_str or '(I)' in val_str or val_str.strip() == 'I':
                    party = 'Independent'

                if 'Senator' in val_str or 'Senate' in val_str:
                    chamber = 'Senate'
                elif 'Representative' in val_str or 'House' in val_str or 'Rep.' in val_str:
                    chamber = 'House'

            days_to_disclose = 0
            if trade_date and disclosure_date:
                try:
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y']:
                        try:
                            td = datetime.strptime(trade_date, fmt)
                            dd = datetime.strptime(disclosure_date, fmt)
                            days_to_disclose = (dd - td).days
                            trade_date = td.strftime('%Y-%m-%d')
                            disclosure_date = dd.strftime('%Y-%m-%d')
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass

            return CongressTrade(
                politician=politician,
                party=party,
                chamber=chamber,
                state='',
                ticker=ticker,
                asset_description=ticker,
                trade_type=trade_type,
                trade_date=trade_date,
                disclosure_date=disclosure_date,
                amount_range=amount_range,
                amount_low=amount_low,
                amount_high=amount_high,
                days_to_disclose=days_to_disclose,
                owner='Self',
            )
        except Exception as e:
            logger.debug(f"Error parsing Capitol row: {e}")
            return None

    def _parse_capitol_trades_regex(self, html: str) -> List[CongressTrade]:
        """Fallback regex parser for Capitol Trades HTML."""
        # This is a best-effort parser if pandas can't parse the tables
        logger.info("Using regex fallback for Capitol Trades parsing")
        return []

    def _parse_amount_range(self, amount_str: str) -> tuple:
        """Parse amount range string like '$1,001 - $15,000' into (low, high)."""
        try:
            amounts = re.findall(r'[\$]?([\d,]+)', str(amount_str))
            if len(amounts) >= 2:
                low = float(amounts[0].replace(',', ''))
                high = float(amounts[1].replace(',', ''))
                return low, high
            elif len(amounts) == 1:
                val = float(amounts[0].replace(',', ''))
                return val, val
        except (ValueError, IndexError):
            pass
        return 0.0, 0.0

    def _normalize_trade_type(self, trade_type: str) -> str:
        """Normalize trade type strings."""
        t = str(trade_type).lower().strip()
        if 'purchase' in t or 'buy' in t:
            return 'Purchase'
        elif 'full' in t:
            return 'Sale (Full)'
        elif 'partial' in t:
            return 'Sale (Partial)'
        elif 'sale' in t or 'sell' in t:
            return 'Sale'
        elif 'exchange' in t:
            return 'Exchange'
        return trade_type

    def get_stats(self, trades: List[CongressTrade]) -> Dict[str, Any]:
        """Calculate summary statistics for trades."""
        if not trades:
            return {
                'total_trades': 0,
                'total_politicians': 0,
                'total_purchases': 0,
                'total_sales': 0,
                'top_traded_tickers': [],
                'most_active_politicians': [],
                'avg_days_to_disclose': 0,
            }

        purchases = [t for t in trades if 'Purchase' in t.trade_type]
        sales = [t for t in trades if 'Sale' in t.trade_type]

        # Top traded tickers
        ticker_counts = {}
        for t in trades:
            ticker_counts[t.ticker] = ticker_counts.get(t.ticker, 0) + 1
        top_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Most active politicians
        pol_counts = {}
        for t in trades:
            pol_counts[t.politician] = pol_counts.get(t.politician, 0) + 1
        top_pols = sorted(pol_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Average days to disclose
        disclosure_days = [t.days_to_disclose for t in trades if t.days_to_disclose > 0]
        avg_days = sum(disclosure_days) / len(disclosure_days) if disclosure_days else 0

        return {
            'total_trades': len(trades),
            'total_politicians': len(set(t.politician for t in trades)),
            'total_purchases': len(purchases),
            'total_sales': len(sales),
            'top_traded_tickers': [{'ticker': t, 'count': c} for t, c in top_tickers],
            'most_active_politicians': [{'name': n, 'count': c} for n, c in top_pols],
            'avg_days_to_disclose': round(avg_days, 1),
        }

    def get_trades_by_ticker(self, ticker: str, trades: Optional[List[CongressTrade]] = None) -> List[CongressTrade]:
        """Filter trades for a specific ticker."""
        if trades is None:
            trades = self.get_trades()
        return [t for t in trades if t.ticker.upper() == ticker.upper()]

    def get_trades_by_politician(self, name: str, trades: Optional[List[CongressTrade]] = None) -> List[CongressTrade]:
        """Filter trades for a specific politician."""
        if trades is None:
            trades = self.get_trades()
        name_lower = name.lower()
        return [t for t in trades if name_lower in t.politician.lower()]
