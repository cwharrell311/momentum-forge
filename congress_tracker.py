"""
Congressional Stock Trade Tracker
Fetches and processes stock trades reported by members of Congress.

Data source priority (tries in order, uses first that works):
  0. Government scraper (free, current, scrapes official EFD/House Clerk sites)
  1. Quiver Quantitative (free, no API key needed)
  2. Finnhub API (free tier available, current data)
  3. Capitol Trades (website scraping, may be blocked)
  4. Stock Watcher S3 buckets (may still have current data)
  5. FMP API (requires paid plan)

NOTE: No fallback to 2020 historical data - we only show current trades.
If no data source works, an empty result is returned with a clear error message.
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import logging
import re
import json
import os

try:
    from gov_scraper import GovScraper
    GOV_SCRAPER_AVAILABLE = True
except ImportError:
    GOV_SCRAPER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Senator party/state lookup for enrichment
SENATOR_PARTIES = {
    'A. Mitchell McConnell': ('Republican', 'KY'),
    'Angus S. King': ('Independent', 'ME'),
    'Benjamin E. Sasse': ('Republican', 'NE'),
    'Benjamin L Cardin': ('Democrat', 'MD'),
    'Bill Cassidy': ('Republican', 'LA'),
    'Bill Hagerty': ('Republican', 'TN'),
    'Charles E. Grassley': ('Republican', 'IA'),
    'Charles E. Schumer': ('Democrat', 'NY'),
    'Chris Van Hollen': ('Democrat', 'MD'),
    'Christopher A Coons': ('Democrat', 'DE'),
    'Christopher A. Coons': ('Democrat', 'DE'),
    'Cindy Hyde-Smith': ('Republican', 'MS'),
    'Cory A. Booker': ('Democrat', 'NJ'),
    'Dan Sullivan': ('Republican', 'AK'),
    'David A. Perdue': ('Republican', 'GA'),
    'Debbie Stabenow': ('Democrat', 'MI'),
    'Dianne Feinstein': ('Democrat', 'CA'),
    'Gary C. Peters': ('Democrat', 'MI'),
    'Jack Reed': ('Democrat', 'RI'),
    'Jacklyn S Rosen': ('Democrat', 'NV'),
    'Jacklyn S. Rosen': ('Democrat', 'NV'),
    'James Lankford': ('Republican', 'OK'),
    'Jerry Moran': ('Republican', 'KS'),
    'Jim Inhofe': ('Republican', 'OK'),
    'Jim Risch': ('Republican', 'ID'),
    'John Barrasso': ('Republican', 'WY'),
    'John Boozman': ('Republican', 'AR'),
    'John Cornyn': ('Republican', 'TX'),
    'John Hickenlooper': ('Democrat', 'CO'),
    'John N Kennedy': ('Republican', 'LA'),
    'John N. Kennedy': ('Republican', 'LA'),
    'Joseph Manchin': ('Democrat', 'WV'),
    'John Hoeven': ('Republican', 'ND'),
    'John Thune': ('Republican', 'SD'),
    'John W. Hickenlooper': ('Democrat', 'CO'),
    'Jon Ossoff': ('Democrat', 'GA'),
    'Kelly Loeffler': ('Republican', 'GA'),
    'Kevin Cramer': ('Republican', 'ND'),
    'Kyrsten Sinema': ('Independent', 'AZ'),
    'Ladda Tammy Duckworth': ('Democrat', 'IL'),
    'Marco Rubio': ('Republican', 'FL'),
    'Maria Cantwell': ('Democrat', 'WA'),
    'Margaret Wood Hassan': ('Democrat', 'NH'),
    'Mark Kelly': ('Democrat', 'AZ'),
    'Mark R. Warner': ('Democrat', 'VA'),
    'Markwayne Mullin': ('Republican', 'OK'),
    'Michael B. Enzi': ('Republican', 'WY'),
    'Michael F. Bennet': ('Democrat', 'CO'),
    'Mike Braun': ('Republican', 'IN'),
    'Mike Crapo': ('Republican', 'ID'),
    'Mike Rounds': ('Republican', 'SD'),
    'Mitt Romney': ('Republican', 'UT'),
    'Nancy Pelosi': ('Democrat', 'CA'),
    'Pat Roberts': ('Republican', 'KS'),
    'Patrick J. Toomey': ('Republican', 'PA'),
    'Patty Murray': ('Democrat', 'WA'),
    'Pete Ricketts': ('Republican', 'NE'),
    'Rafael E Cruz': ('Republican', 'TX'),
    'Rafael E. Cruz': ('Republican', 'TX'),
    'Rand Paul': ('Republican', 'KY'),
    'Raphael G. Warnock': ('Democrat', 'GA'),
    'Richard Blumenthal': ('Democrat', 'CT'),
    'Richard Burr': ('Republican', 'NC'),
    'Rick Scott': ('Republican', 'FL'),
    'Robert Menendez': ('Democrat', 'NJ'),
    'Robert P. Casey': ('Democrat', 'PA'),
    'Roger F. Wicker': ('Republican', 'MS'),
    'Ron L Wyden': ('Democrat', 'OR'),
    'Ron Johnson': ('Republican', 'WI'),
    'Roy Blunt': ('Republican', 'MO'),
    'Sheldon Whitehouse': ('Democrat', 'RI'),
    'Shelley Moore Capito': ('Republican', 'WV'),
    'Steve Daines': ('Republican', 'MT'),
    'Susan M. Collins': ('Republican', 'ME'),
    'Tammy Duckworth': ('Democrat', 'IL'),
    'Ted Cruz': ('Republican', 'TX'),
    'Thomas R. Carper': ('Democrat', 'DE'),
    'Thomas R. Tillis': ('Republican', 'NC'),
    'Thomas Udall': ('Democrat', 'NM'),
    'Timothy M Kaine': ('Democrat', 'VA'),
    'Timothy M. Kaine': ('Democrat', 'VA'),
    'Tim Scott': ('Republican', 'SC'),
    'Todd Young': ('Republican', 'IN'),
    'Tommy Tuberville': ('Republican', 'AL'),
    'Tina Smith': ('Democrat', 'MN'),
    'William Cassidy': ('Republican', 'LA'),
    'William F. Hagerty': ('Republican', 'TN'),
}


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


# Browser-like headers to avoid blocks
BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/html, */*',
    'Accept-Language': 'en-US,en;q=0.9',
}


class CongressTracker:
    """
    Tracks congressional stock trades from multiple data sources.
    Only returns CURRENT data - no fallback to stale 2020 data.
    """

    FMP_BASE_URL = "https://financialmodelingprep.com/api/v4"
    FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

    # S3 buckets that may still have current data
    SENATE_S3_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"
    HOUSE_S3_URL = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"

    def __init__(self, fmp_api_key: Optional[str] = None, finnhub_api_key: Optional[str] = None):
        self.fmp_api_key = fmp_api_key
        self.finnhub_api_key = finnhub_api_key or os.environ.get('FINNHUB_API_KEY')
        self._cache = None
        self._cache_time = None
        self._cache_ttl = 3600  # Cache for 1 hour
        self._gov_scraper = GovScraper() if GOV_SCRAPER_AVAILABLE else None
        self._source_used = None
        self._source_errors = []

    @property
    def source_used(self) -> Optional[str]:
        """Return the data source that was used."""
        return self._source_used

    @property
    def source_errors(self) -> List[str]:
        """Return any errors encountered during data fetch."""
        return self._source_errors

    def get_trades(self, days_back: int = 365) -> List[CongressTrade]:
        """
        Fetch congressional trades from available sources.
        Only returns CURRENT data - if all sources fail, returns empty list.

        Priority:
        0. Government scraper (official EFD sites)
        1. Finnhub API (free tier, current data)
        2. Capitol Trades (website)
        3. Stock Watcher S3 buckets
        4. FMP API (paid)
        """
        # Return cache if fresh
        if self._cache is not None and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < self._cache_ttl:
                logger.info(f"Returning {len(self._cache)} cached trades (source: {self._source_used})")
                return self._cache

        trades = []
        self._source_used = None
        self._source_errors = []

        # Source 0: Government scraper (official sources)
        if self._gov_scraper:
            logger.info(">>> Trying Source 0: Government EFD Scraper...")
            try:
                trades = self._fetch_gov_scraper(days_back)
                if trades:
                    self._source_used = "Government EFD Scraper (official)"
                    logger.info(f"✓ Government scraper: {len(trades)} trades")
            except Exception as e:
                self._source_errors.append(f"Gov scraper: {e}")
                logger.error(f"✗ Government scraper failed: {e}")
        else:
            self._source_errors.append("Gov scraper: not available (import failed)")
            logger.warning("✗ Government scraper not available")

        # Source 1: Quiver Quantitative (free, no API key needed)
        if not trades:
            logger.info(">>> Trying Source 1: Quiver Quantitative...")
            try:
                trades = self._fetch_quiver_trades()
                if trades:
                    self._source_used = "Quiver Quantitative (free)"
                    logger.info(f"✓ Quiver: {len(trades)} trades")
            except Exception as e:
                self._source_errors.append(f"Quiver: {e}")
                logger.error(f"✗ Quiver failed: {e}")

        # Source 2: Finnhub API (free tier available)
        if not trades and self.finnhub_api_key:
            logger.info(">>> Trying Source 2: Finnhub API...")
            try:
                trades = self._fetch_finnhub_trades(days_back)
                if trades:
                    self._source_used = "Finnhub API (free tier)"
                    logger.info(f"✓ Finnhub: {len(trades)} trades")
            except Exception as e:
                self._source_errors.append(f"Finnhub: {e}")
                logger.error(f"✗ Finnhub failed: {e}")
        elif not trades:
            self._source_errors.append("Finnhub: no API key (set FINNHUB_API_KEY env var)")
            logger.info("✗ Finnhub: no API key set")

        # Source 3: Capitol Trades website
        if not trades:
            logger.info(">>> Trying Source 3: Capitol Trades website...")
            try:
                trades = self._fetch_capitol_trades()
                if trades:
                    self._source_used = "Capitol Trades"
                    logger.info(f"✓ Capitol Trades: {len(trades)} trades")
            except Exception as e:
                self._source_errors.append(f"Capitol Trades: {e}")
                logger.error(f"✗ Capitol Trades failed: {e}")

        # Source 4: Stock Watcher S3 buckets (check if they have current data)
        if not trades:
            logger.info(">>> Trying Source 4: Stock Watcher S3 buckets...")
            try:
                trades = self._fetch_stock_watcher_s3()
                if trades:
                    # Check if data is current (within last year)
                    newest_date = max(t.trade_date for t in trades if t.trade_date)
                    try:
                        newest_dt = datetime.strptime(newest_date, '%Y-%m-%d')
                        days_old = (datetime.now() - newest_dt).days
                        if days_old > 365:
                            logger.warning(f"S3 data is {days_old} days old - discarding")
                            self._source_errors.append(f"S3 buckets: data too old ({days_old} days)")
                            trades = []
                        else:
                            self._source_used = f"Stock Watcher S3 (newest: {newest_date})"
                            logger.info(f"✓ S3 buckets: {len(trades)} trades (newest: {newest_date})")
                    except ValueError:
                        # Can't parse date, accept anyway
                        self._source_used = "Stock Watcher S3"
                        logger.info(f"✓ S3 buckets: {len(trades)} trades")
            except Exception as e:
                self._source_errors.append(f"S3 buckets: {e}")
                logger.error(f"✗ S3 buckets failed: {e}")

        # Source 5: FMP API (paid)
        if not trades and self.fmp_api_key:
            logger.info(">>> Trying Source 5: FMP API...")
            try:
                trades = self._fetch_fmp_trades(days_back)
                if trades:
                    self._source_used = "FMP API (paid)"
                    logger.info(f"✓ FMP: {len(trades)} trades")
            except Exception as e:
                self._source_errors.append(f"FMP: {e}")
                logger.error(f"✗ FMP failed: {e}")
        elif not trades:
            self._source_errors.append("FMP: no API key or free tier (403)")
            logger.info("✗ FMP: no API key")

        # NO FALLBACK TO 2020 DATA - we only want current trades!
        if not trades:
            logger.error("=" * 60)
            logger.error("ALL DATA SOURCES FAILED - NO CURRENT CONGRESSIONAL TRADE DATA")
            logger.error("Errors:")
            for err in self._source_errors:
                logger.error(f"  - {err}")
            logger.error("")
            logger.error("To fix this:")
            logger.error("  1. Get a free Finnhub API key at https://finnhub.io/register")
            logger.error("     Then set: export FINNHUB_API_KEY=your_key")
            logger.error("  2. OR upgrade FMP to a paid plan (~$19/mo)")
            logger.error("=" * 60)
            self._source_used = "NONE - All sources failed"

        # Sort by trade date (newest first)
        if trades:
            trades.sort(key=lambda t: t.trade_date, reverse=True)

        # Cache results
        self._cache = trades
        self._cache_time = datetime.now()

        logger.info(f"Final source: {self._source_used} | Total trades: {len(trades)}")
        return trades

    # ------------------------------------------------------------------
    # Source 0: Government scraper (official disclosure sites)
    # ------------------------------------------------------------------
    def _fetch_gov_scraper(self, days_back: int) -> List[CongressTrade]:
        """Scrape official government disclosure sites for current trades."""
        trades = []
        raw_trades = self._gov_scraper.scrape_all(days_back=days_back, use_cache=True)

        for item in raw_trades:
            trade = self._parse_gov_trade(item)
            if trade:
                trades.append(trade)

        logger.info(f"Government scraper: {len(trades)} trades parsed")
        return trades

    # ------------------------------------------------------------------
    # Source 1: Quiver Quantitative (free, no API key needed)
    # ------------------------------------------------------------------
    def _fetch_quiver_trades(self) -> List[CongressTrade]:
        """
        Fetch congressional trades from Quiver Quantitative.
        Free public data, no API key required.
        """
        trades = []

        # Quiver has a public-facing page with congressional trading data
        urls_to_try = [
            "https://www.quiverquant.com/congresstrading/",
            "https://api.quiverquant.com/beta/live/congresstrading",
        ]

        for url in urls_to_try:
            try:
                logger.info(f"Trying Quiver URL: {url}")
                resp = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
                logger.info(f"Quiver response: {resp.status_code}")

                if resp.status_code != 200:
                    continue

                # Check if it's JSON (API) or HTML (website)
                content_type = resp.headers.get('content-type', '')

                if 'json' in content_type:
                    data = resp.json()
                    if isinstance(data, list):
                        for item in data:
                            trade = self._parse_quiver_trade(item)
                            if trade:
                                trades.append(trade)
                elif 'html' in content_type:
                    # Try to extract __NEXT_DATA__ or similar
                    trades = self._parse_quiver_html(resp.text)

                if trades:
                    logger.info(f"Quiver: {len(trades)} trades from {url}")
                    break

            except Exception as e:
                logger.debug(f"Quiver URL {url} failed: {e}")
                continue

        return trades

    def _parse_quiver_trade(self, item: Dict) -> Optional[CongressTrade]:
        """Parse a Quiver Quantitative trade item."""
        try:
            ticker = item.get('Ticker', item.get('ticker', ''))
            if not ticker:
                return None

            name = item.get('Representative', item.get('Name', item.get('name', 'Unknown')))
            tx_date = item.get('TransactionDate', item.get('transaction_date', ''))
            disclosure_date = item.get('ReportDate', item.get('disclosure_date', tx_date))
            tx_type = item.get('Transaction', item.get('type', 'Unknown'))
            amount = item.get('Range', item.get('amount', ''))
            party = item.get('Party', item.get('party', 'Unknown'))
            chamber = item.get('House', item.get('chamber', 'House'))
            if chamber in ('House', 'house', 'H'):
                chamber = 'House'
            else:
                chamber = 'Senate'

            amount_low, amount_high = self._parse_amount_range(amount)

            return CongressTrade(
                politician=name,
                party=self._normalize_party(party),
                chamber=chamber,
                state='',
                ticker=ticker.upper(),
                asset_description=item.get('Description', ticker),
                trade_type=self._normalize_trade_type(tx_type),
                trade_date=self._standardize_date(tx_date),
                disclosure_date=self._standardize_date(disclosure_date),
                amount_range=amount,
                amount_low=amount_low,
                amount_high=amount_high,
                days_to_disclose=0,
                owner='Self',
            )
        except Exception as e:
            logger.debug(f"Error parsing Quiver trade: {e}")
            return None

    def _parse_quiver_html(self, html: str) -> List[CongressTrade]:
        """Extract trade data from Quiver Quantitative HTML page."""
        trades = []
        try:
            # Look for Next.js data or embedded JSON
            import re
            match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
            if match:
                next_data = json.loads(match.group(1))
                page_props = next_data.get('props', {}).get('pageProps', {})
                trade_data = page_props.get('trades', page_props.get('data', []))

                if isinstance(trade_data, list):
                    for item in trade_data:
                        trade = self._parse_quiver_trade(item)
                        if trade:
                            trades.append(trade)
        except Exception as e:
            logger.debug(f"Quiver HTML parse error: {e}")

        return trades

    # ------------------------------------------------------------------
    # Source 2: Finnhub API (free tier available)
    # ------------------------------------------------------------------
    def _fetch_finnhub_trades(self, days_back: int) -> List[CongressTrade]:
        """
        Fetch congressional trades from Finnhub API.
        Free tier: 60 calls/minute, congressional trading endpoint.
        """
        trades = []

        # Finnhub endpoint: /stock/congressional-trading
        # Requires: symbol, from, to dates
        # We'll query for popular stocks that Congress frequently trades
        popular_tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'TSLA', 'JPM', 'BAC', 'WFC',
                          'XOM', 'CVX', 'PFE', 'JNJ', 'UNH', 'V', 'MA', 'HD', 'DIS', 'NFLX']

        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')

        for ticker in popular_tickers:
            try:
                url = f"{self.FINNHUB_BASE_URL}/stock/congressional-trading"
                params = {
                    'symbol': ticker,
                    'from': from_date,
                    'to': to_date,
                    'token': self.finnhub_api_key,
                }

                resp = requests.get(url, params=params, headers=BROWSER_HEADERS, timeout=15)

                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict) and 'data' in data:
                        items = data['data']
                    elif isinstance(data, list):
                        items = data
                    else:
                        continue

                    for item in items:
                        trade = self._parse_finnhub_trade(item, ticker)
                        if trade:
                            trades.append(trade)

                elif resp.status_code == 403:
                    logger.warning(f"Finnhub: 403 Forbidden (may need premium for this endpoint)")
                    return []  # Don't keep trying if blocked
                elif resp.status_code == 429:
                    logger.warning("Finnhub: Rate limited, stopping")
                    break

            except Exception as e:
                logger.debug(f"Finnhub error for {ticker}: {e}")

        return trades

    def _parse_finnhub_trade(self, item: Dict, default_ticker: str) -> Optional[CongressTrade]:
        """Parse a Finnhub congressional trade item."""
        try:
            ticker = item.get('symbol', default_ticker)
            if not ticker:
                return None

            name = item.get('name', item.get('filingName', 'Unknown'))
            tx_date = item.get('transactionDate', '')
            filing_date = item.get('filingDate', '')
            tx_type = item.get('transactionType', 'Unknown')
            amount = item.get('amountFrom', 0)
            amount_to = item.get('amountTo', amount)

            # Determine party/chamber from name or other fields
            party = item.get('party', 'Unknown')
            chamber = 'Senate' if 'senator' in name.lower() else 'House'

            amount_str = f"${amount:,.0f} - ${amount_to:,.0f}" if amount else ''

            return CongressTrade(
                politician=name,
                party=party,
                chamber=chamber,
                state='',
                ticker=ticker.upper(),
                asset_description=item.get('assetName', ticker),
                trade_type=self._normalize_trade_type(tx_type),
                trade_date=self._standardize_date(tx_date),
                disclosure_date=self._standardize_date(filing_date),
                amount_range=amount_str,
                amount_low=float(amount) if amount else 0.0,
                amount_high=float(amount_to) if amount_to else 0.0,
                days_to_disclose=0,
                owner=item.get('ownerType', 'Self'),
            )
        except Exception as e:
            logger.debug(f"Error parsing Finnhub trade: {e}")
            return None

    def _parse_gov_trade(self, item: Dict) -> Optional[CongressTrade]:
        """Parse a raw transaction dict from the government scraper."""
        try:
            ticker = item.get('ticker', '').strip()
            if not ticker or ticker in ('--', 'N/A', ''):
                return None

            politician = item.get('politician', 'Unknown')
            party, state = self._lookup_senator_info(politician)
            if item.get('state'):
                state = item['state']

            trade_date = self._standardize_date(item.get('trade_date', ''))
            disclosure_date = self._standardize_date(item.get('disclosure_date', ''))
            if not disclosure_date:
                disclosure_date = trade_date

            days_to_disclose = 0
            if trade_date and disclosure_date and trade_date != disclosure_date:
                try:
                    td = datetime.strptime(trade_date, '%Y-%m-%d')
                    dd = datetime.strptime(disclosure_date, '%Y-%m-%d')
                    days_to_disclose = max(0, (dd - td).days)
                except ValueError:
                    pass

            amount_str = item.get('amount', '$0')
            amount_low, amount_high = self._parse_amount_range(amount_str)

            return CongressTrade(
                politician=politician,
                party=party,
                chamber=item.get('chamber', 'Senate'),
                state=state,
                ticker=ticker.upper(),
                asset_description=item.get('asset_description', ticker),
                trade_type=self._normalize_trade_type(item.get('trade_type', 'Unknown')),
                trade_date=trade_date,
                disclosure_date=disclosure_date,
                amount_range=amount_str,
                amount_low=amount_low,
                amount_high=amount_high,
                days_to_disclose=days_to_disclose,
                owner=item.get('owner', 'Self'),
            )
        except Exception as e:
            logger.debug(f"Error parsing gov trade: {e}")
            return None

    # ------------------------------------------------------------------
    # Source 3: Capitol Trades (capitoltrades.com)
    # ------------------------------------------------------------------
    def _fetch_capitol_trades(self) -> List[CongressTrade]:
        """
        Fetch from Capitol Trades. They have a public-facing site with trade data.
        Tries their internal API endpoint used by the frontend.
        """
        trades = []
        try:
            # Capitol Trades uses a Next.js frontend with API routes
            url = "https://www.capitoltrades.com/trades?page=1&pageSize=96"
            logger.info("Trying Capitol Trades...")
            response = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
            logger.info(f"Capitol Trades response: {response.status_code}")

            if response.status_code != 200:
                return trades

            # If we get HTML, try to extract the __NEXT_DATA__ JSON
            text = response.text
            if '<!DOCTYPE' in text or '<html' in text:
                trades = self._parse_capitol_trades_html(text)
            else:
                # Direct JSON response
                try:
                    data = response.json()
                    trades = self._parse_capitol_trades_json(data)
                except json.JSONDecodeError:
                    pass

            if trades:
                logger.info(f"Capitol Trades: {len(trades)} trades fetched")

        except Exception as e:
            logger.info(f"Capitol Trades unavailable: {e}")

        return trades

    def _parse_capitol_trades_html(self, html: str) -> List[CongressTrade]:
        """Extract trade data from Capitol Trades HTML page using __NEXT_DATA__."""
        trades = []
        try:
            # Look for Next.js data payload
            match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
            if not match:
                logger.info("Capitol Trades: No __NEXT_DATA__ found, trying direct HTML parse")
                # Try direct HTML table parsing as fallback
                trades = self._parse_capitol_trades_table(html)
                return trades

            next_data = json.loads(match.group(1))
            # Navigate the Next.js data structure - try multiple paths
            page_props = next_data.get('props', {}).get('pageProps', {})

            # Log available keys for debugging
            logger.info(f"Capitol Trades pageProps keys: {list(page_props.keys())[:10]}")

            trade_data = page_props.get('trades', page_props.get('data', page_props.get('tradeList', [])))

            if isinstance(trade_data, dict):
                trade_data = trade_data.get('data', trade_data.get('trades', trade_data.get('results', [])))

            logger.info(f"Capitol Trades: Found {len(trade_data) if isinstance(trade_data, list) else 0} items in JSON")

            for item in trade_data:
                trade = self._parse_capitol_trade_item(item)
                if trade:
                    trades.append(trade)

        except (json.JSONDecodeError, KeyError) as e:
            logger.info(f"Failed to parse Capitol Trades JSON: {e}, trying HTML table")

        return trades

    def _parse_capitol_trades_table(self, html: str) -> List[CongressTrade]:
        """Fallback: Parse Capitol Trades HTML table directly."""
        trades = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Log a sample of the HTML structure for debugging
            sample_divs = soup.find_all('div', limit=5)
            if sample_divs and not hasattr(self, '_logged_capitol_sample'):
                logger.info(f"Capitol Trades sample HTML structure: {str(sample_divs[0])[:300]}")
                self._logged_capitol_sample = True

            # Log raw HTML sample IMMEDIATELY for debugging
            html_sample = html[:1000] if len(html) > 1000 else html
            logger.info(f"Capitol Trades raw HTML sample: {html_sample[:500]}")

            # Look for trade rows - Capitol Trades uses various table/list structures
            rows = soup.find_all('tr') or soup.find_all('div', class_=re.compile(r'trade|row', re.I))
            logger.info(f"Capitol Trades found {len(rows)} rows to parse")

            for row in rows[:100]:  # Limit
                cells = row.find_all(['td', 'span', 'div'])
                if len(cells) < 3:
                    continue

                ticker = ''
                name = ''
                party = 'Unknown'
                chamber = 'Unknown'
                state = ''
                tx_type = ''
                tx_date = ''
                amount = ''

                for cell in cells:
                    text = cell.get_text(strip=True)

                    # Ticker pattern (1-5 uppercase letters, not common words)
                    if re.match(r'^[A-Z]{1,5}$', text) and text not in ('BUY', 'SELL', 'USD', 'THE', 'FOR', 'AND'):
                        if not ticker:  # Only take first ticker found
                            ticker = text

                    # Date patterns - try multiple formats
                    # Format: Jan 15, 2026 or January 15, 2026
                    elif re.match(r'^[A-Z][a-z]{2,8}\s+\d{1,2},?\s+\d{4}$', text):
                        tx_date = text
                    # Format: 01/15/2026 or 1/15/26
                    elif re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', text):
                        tx_date = text
                    # Format: 2026-01-15
                    elif re.match(r'^\d{4}-\d{2}-\d{2}$', text):
                        tx_date = text

                    # Trade type
                    elif text.lower() in ('buy', 'sell', 'purchase', 'sale', 'sold', 'bought'):
                        tx_type = text

                    # Amount (contains $)
                    elif '$' in text and not amount:
                        amount = text

                    # Party detection
                    elif text in ('Democrat', 'Republican', 'Independent', 'D', 'R', 'I'):
                        party = text

                    # Chamber detection
                    elif text in ('House', 'Senate', 'H', 'S'):
                        chamber = text

                    # State (2 letter code)
                    elif re.match(r'^[A-Z]{2}$', text) and text not in ('US', 'NA', 'OK'):
                        state = text

                    # Name - longer text that looks like a name (not containing common non-name patterns)
                    elif len(text) > 3 and not name:
                        # Check if it's a concatenated name like "Kelly MorrisonDemocratHouseMN"
                        # Try to split it
                        concat_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(Democrat|Republican|Independent)?(House|Senate)?([A-Z]{2})?$', text)
                        if concat_match:
                            name = concat_match.group(1).strip()
                            if concat_match.group(2):
                                party = concat_match.group(2)
                            if concat_match.group(3):
                                chamber = concat_match.group(3)
                            if concat_match.group(4):
                                state = concat_match.group(4)
                        elif not any(skip in text.lower() for skip in ['purchase', 'sale', 'buy', 'sell', '$', 'total']):
                            name = text

                # If no date found yet, search the entire row text for date patterns
                if ticker and not tx_date:
                    row_text = row.get_text()
                    # Try various date patterns in the full row text
                    date_patterns = [
                        r'(\d{1,2}/\d{1,2}/\d{2,4})',  # 01/15/2026
                        r'(\d{4}-\d{2}-\d{2})',  # 2026-01-15
                        r'([A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4})',  # Jan 15, 2026
                        r'([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',  # January 15, 2026
                    ]
                    for pattern in date_patterns:
                        date_match = re.search(pattern, row_text)
                        if date_match:
                            tx_date = date_match.group(1)
                            break

                # Log first few rows to debug
                if ticker and len(trades) < 3:
                    row_text_sample = row.get_text()[:200].replace('\n', ' ')
                    logger.info(f"Capitol row text: {row_text_sample}")
                    logger.info(f"  Extracted: ticker={ticker}, date='{tx_date}', name={name[:30] if name else 'none'}")

                if ticker:
                    amount_low, amount_high = self._parse_amount_range(amount)
                    standardized_date = self._standardize_date(tx_date)
                    trades.append(CongressTrade(
                        politician=name or 'Unknown',
                        party=self._normalize_party(party),
                        chamber=self._normalize_chamber(chamber),
                        state=state,
                        ticker=ticker,
                        asset_description=ticker,
                        trade_type=self._normalize_trade_type(tx_type),
                        trade_date=standardized_date,
                        disclosure_date=standardized_date,
                        amount_range=amount,
                        amount_low=amount_low,
                        amount_high=amount_high,
                        days_to_disclose=0,
                        owner='Self',
                    ))

            logger.info(f"Capitol Trades HTML table: {len(trades)} trades parsed")
        except Exception as e:
            logger.debug(f"Capitol Trades table parse error: {e}")

        return trades

    def _parse_capitol_trades_json(self, data: Any) -> List[CongressTrade]:
        """Parse Capitol Trades JSON API response."""
        trades = []
        items = data if isinstance(data, list) else data.get('data', data.get('trades', []))
        for item in items:
            trade = self._parse_capitol_trade_item(item)
            if trade:
                trades.append(trade)
        return trades

    def _parse_capitol_trade_item(self, item: Dict) -> Optional[CongressTrade]:
        """Parse a single Capitol Trades trade item."""
        try:
            # Capitol Trades has nested politician/asset objects
            politician_data = item.get('politician', {})
            if isinstance(politician_data, dict):
                name = f"{politician_data.get('firstName', '')} {politician_data.get('lastName', '')}".strip()
                party = politician_data.get('party', 'Unknown')
                chamber = politician_data.get('chamber', 'Unknown')
                state = politician_data.get('state', '')
            else:
                name = item.get('politician', item.get('politicianName', 'Unknown'))
                party = item.get('party', 'Unknown')
                chamber = item.get('chamber', 'Unknown')
                state = item.get('state', '')

            asset_data = item.get('asset', {})
            if isinstance(asset_data, dict):
                ticker = asset_data.get('ticker', asset_data.get('assetTicker', ''))
                asset_desc = asset_data.get('name', asset_data.get('assetName', ticker))
            else:
                ticker = item.get('ticker', item.get('assetTicker', ''))
                asset_desc = item.get('assetName', ticker)

            if not ticker or ticker == '--':
                return None

            trade_date = item.get('txDate', item.get('transactionDate', ''))
            disclosure_date = item.get('filingDate', item.get('disclosureDate', ''))

            # Parse dates to standard format
            trade_date = self._standardize_date(trade_date)
            disclosure_date = self._standardize_date(disclosure_date)

            days_to_disclose = 0
            if trade_date and disclosure_date:
                try:
                    td = datetime.strptime(trade_date, '%Y-%m-%d')
                    dd = datetime.strptime(disclosure_date, '%Y-%m-%d')
                    days_to_disclose = (dd - td).days
                except ValueError:
                    pass

            amount = item.get('amount', item.get('range', '$0'))
            amount_low, amount_high = self._parse_amount_range(amount)
            trade_type = self._normalize_trade_type(item.get('txType', item.get('type', 'Unknown')))

            return CongressTrade(
                politician=name or 'Unknown',
                party=self._normalize_party(party),
                chamber=self._normalize_chamber(chamber),
                state=state,
                ticker=ticker.upper(),
                asset_description=asset_desc,
                trade_type=trade_type,
                trade_date=trade_date,
                disclosure_date=disclosure_date,
                amount_range=amount if isinstance(amount, str) else f"${amount_low:,.0f} - ${amount_high:,.0f}",
                amount_low=amount_low,
                amount_high=amount_high,
                days_to_disclose=max(0, days_to_disclose),
                owner=item.get('owner', 'Self'),
            )
        except Exception as e:
            logger.debug(f"Error parsing Capitol Trade item: {e}")
            return None

    # ------------------------------------------------------------------
    # Source 4: Stock Watcher S3 buckets (check for current data)
    # ------------------------------------------------------------------
    def _fetch_stock_watcher_s3(self) -> List[CongressTrade]:
        """
        Fetch from S3 buckets that may still have current data.
        These are updated by senatestockwatcher.com / housestockwatcher.com.
        """
        trades = []

        # Senate S3 bucket
        try:
            logger.info(f"Fetching Senate S3 bucket...")
            resp = requests.get(self.SENATE_S3_URL, headers=BROWSER_HEADERS, timeout=60)
            logger.info(f"Senate S3 response: {resp.status_code}")

            if resp.status_code == 200:
                data = resp.json()
                for item in data:
                    trade = self._parse_stock_watcher_trade(item, 'Senate')
                    if trade:
                        trades.append(trade)
                logger.info(f"Senate S3: {len(trades)} trades parsed")

        except Exception as e:
            logger.warning(f"Senate S3 failed: {e}")

        # House S3 bucket
        house_count = 0
        try:
            logger.info(f"Fetching House S3 bucket...")
            resp = requests.get(self.HOUSE_S3_URL, headers=BROWSER_HEADERS, timeout=60)
            logger.info(f"House S3 response: {resp.status_code}")

            if resp.status_code == 200:
                data = resp.json()
                for item in data:
                    trade = self._parse_stock_watcher_trade(item, 'House')
                    if trade:
                        trades.append(trade)
                        house_count += 1
                logger.info(f"House S3: {house_count} trades parsed")

        except Exception as e:
            logger.warning(f"House S3 failed: {e}")

        return trades

    def _parse_stock_watcher_trade(self, item: Dict, chamber: str) -> Optional[CongressTrade]:
        """Parse a trade from Senate/House Stock Watcher APIs."""
        try:
            ticker = item.get('ticker', item.get('symbol', '')).strip()
            if not ticker or ticker == '--' or ticker == 'N/A':
                return None

            # Senate uses 'senator', House uses 'representative'
            politician = item.get('senator', item.get('representative', item.get('name', 'Unknown'))).strip()
            party, state = self._lookup_senator_info(politician)

            raw_date = item.get('transaction_date', item.get('transactionDate', ''))
            trade_date = self._standardize_date(raw_date)

            disclosure_date = item.get('disclosure_date', item.get('disclosureDate', ''))
            disclosure_date = self._standardize_date(disclosure_date) if disclosure_date else trade_date

            days_to_disclose = 0
            if trade_date and disclosure_date and trade_date != disclosure_date:
                try:
                    td = datetime.strptime(trade_date, '%Y-%m-%d')
                    dd = datetime.strptime(disclosure_date, '%Y-%m-%d')
                    days_to_disclose = (dd - td).days
                except ValueError:
                    pass

            amount_str = item.get('amount', '$0 - $0')
            amount_low, amount_high = self._parse_amount_range(amount_str)

            return CongressTrade(
                politician=politician,
                party=item.get('party', party),
                chamber=chamber,
                state=item.get('state', state),
                ticker=ticker.upper(),
                asset_description=item.get('asset_description', item.get('assetDescription', ticker)),
                trade_type=self._normalize_trade_type(item.get('type', item.get('transaction_type', 'Unknown'))),
                trade_date=trade_date,
                disclosure_date=disclosure_date,
                amount_range=amount_str,
                amount_low=amount_low,
                amount_high=amount_high,
                days_to_disclose=max(0, days_to_disclose),
                owner=item.get('owner', 'Self').strip() if item.get('owner', '--') != '--' else 'Self',
            )
        except Exception as e:
            logger.debug(f"Error parsing stock watcher trade: {e}")
            return None

    # ------------------------------------------------------------------
    # Source 5: FMP API (paid)
    # ------------------------------------------------------------------
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
                logger.info(f"FMP {chamber} trading API: {response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        for item in data:
                            trade = self._parse_fmp_trade(item, chamber)
                            if trade:
                                trades.append(trade)
                        logger.info(f"FMP: {len(data)} {chamber} trades")
                    else:
                        return []
                elif response.status_code == 403:
                    logger.info(f"FMP {chamber} endpoint paywalled (403)")
                    return []

            except Exception as e:
                logger.error(f"FMP {chamber} error: {e}")

        return trades

    def _parse_fmp_trade(self, item: Dict, chamber: str) -> Optional[CongressTrade]:
        """Parse a single FMP trade entry."""
        try:
            ticker = item.get('ticker', item.get('symbol', ''))
            if not ticker or ticker == '--' or ticker == 'N/A':
                return None

            trade_date = item.get('transactionDate', item.get('transaction_date', ''))
            disclosure_date = item.get('disclosureDate', item.get('disclosure_date', ''))

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

            politician = item.get('firstName', item.get('representative', ''))
            last_name = item.get('lastName', '')
            if last_name:
                politician = f"{politician} {last_name}".strip()

            return CongressTrade(
                politician=politician or 'Unknown',
                party=item.get('party', 'Unknown'),
                chamber=chamber,
                state=item.get('state', item.get('district', '')),
                ticker=ticker.upper(),
                asset_description=item.get('assetDescription', item.get('asset_description', ticker)),
                trade_type=self._normalize_trade_type(item.get('type', item.get('transaction', 'Unknown'))),
                trade_date=trade_date,
                disclosure_date=disclosure_date,
                amount_range=amount,
                amount_low=amount_low,
                amount_high=amount_high,
                days_to_disclose=max(0, days_to_disclose),
                owner=item.get('owner', 'Self'),
            )
        except Exception as e:
            logger.debug(f"Error parsing FMP trade: {e}")
            return None

    # ------------------------------------------------------------------
    # NOTE: No historical GitHub fallback - we only want current data
    # The old senate-stock-watcher-data GitHub repo hasn't been updated
    # since March 2021 and contains data from 2012-2020.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _standardize_date(self, date_str: str) -> str:
        """Convert various date formats to YYYY-MM-DD."""
        if not date_str:
            return ''
        date_str = str(date_str).strip()

        # Already in YYYY-MM-DD
        if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
            return date_str[:10]

        # MM/DD/YYYY
        try:
            dt = datetime.strptime(date_str, '%m/%d/%Y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass

        # M/D/YYYY (single digit month/day)
        try:
            dt = datetime.strptime(date_str, '%m/%d/%Y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass

        # MM/DD/YY (2-digit year)
        try:
            dt = datetime.strptime(date_str, '%m/%d/%y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass

        # Jan 15, 2026 or Jan 15 2026
        try:
            # Remove comma if present
            clean_date = date_str.replace(',', '')
            dt = datetime.strptime(clean_date, '%b %d %Y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass

        # January 15, 2026 or January 15 2026
        try:
            clean_date = date_str.replace(',', '')
            dt = datetime.strptime(clean_date, '%B %d %Y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass

        # 15 Jan 2026
        try:
            dt = datetime.strptime(date_str, '%d %b %Y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass

        # YYYY-MM-DDTHH:MM:SS (ISO format)
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except (ValueError, AttributeError):
            pass

        return date_str

    def _lookup_senator_info(self, senator_name: str) -> tuple:
        """Look up party and state for a senator by name."""
        if senator_name in SENATOR_PARTIES:
            return SENATOR_PARTIES[senator_name]

        # Normalize: remove periods, commas, Jr/Sr/III suffixes, extra spaces
        normalized = re.sub(r'[.,]', '', senator_name)
        normalized = re.sub(r'\s+(Jr|Sr|III|II|IV)\s*$', '', normalized, flags=re.IGNORECASE)
        normalized = ' '.join(normalized.split())

        for known_name, info in SENATOR_PARTIES.items():
            known_normalized = re.sub(r'[.,]', '', known_name)
            known_normalized = ' '.join(known_normalized.split())
            if normalized.lower() == known_normalized.lower():
                return info

        # Partial matching (last name + first initial)
        name_parts = normalized.split()
        if name_parts:
            last_name = name_parts[-1].lower()
            for known_name, (party, state) in SENATOR_PARTIES.items():
                known_parts = known_name.replace('.', '').split()
                known_last = known_parts[-1].lower()
                if last_name == known_last:
                    if len(name_parts) > 1 and len(known_parts) > 1:
                        if name_parts[0][0].lower() == known_parts[0][0].lower():
                            return (party, state)

        return ('Unknown', '')

    def _parse_amount_range(self, amount_str: str) -> tuple:
        """Parse amount range string like '$1,001 - $15,000' into (low, high)."""
        try:
            over_match = re.search(r'Over\s+\$?([\d,]+)', str(amount_str), re.IGNORECASE)
            if over_match:
                val = float(over_match.group(1).replace(',', ''))
                return val, val * 2

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

    def _normalize_party(self, party: str) -> str:
        """Normalize party names."""
        p = str(party).lower().strip()
        if p in ('d', 'dem', 'democrat', 'democratic'):
            return 'Democrat'
        elif p in ('r', 'rep', 'republican', 'gop'):
            return 'Republican'
        elif p in ('i', 'ind', 'independent'):
            return 'Independent'
        return party

    def _normalize_chamber(self, chamber: str) -> str:
        """Normalize chamber names."""
        c = str(chamber).lower().strip()
        if 'senate' in c or c == 's':
            return 'Senate'
        elif 'house' in c or c == 'h':
            return 'House'
        return chamber

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
                'data_source': self._source_used or 'None',
                'source_errors': self._source_errors,
            }

        purchases = [t for t in trades if 'Purchase' in t.trade_type]
        sales = [t for t in trades if 'Sale' in t.trade_type]

        ticker_counts = {}
        for t in trades:
            ticker_counts[t.ticker] = ticker_counts.get(t.ticker, 0) + 1
        top_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        pol_counts = {}
        for t in trades:
            pol_counts[t.politician] = pol_counts.get(t.politician, 0) + 1
        top_pols = sorted(pol_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        disclosure_days = [t.days_to_disclose for t in trades if t.days_to_disclose > 0]
        avg_days = sum(disclosure_days) / len(disclosure_days) if disclosure_days else 0

        # Get date range
        dates = [t.trade_date for t in trades if t.trade_date]
        newest_date = max(dates) if dates else ''
        oldest_date = min(dates) if dates else ''

        return {
            'total_trades': len(trades),
            'total_politicians': len(set(t.politician for t in trades)),
            'total_purchases': len(purchases),
            'total_sales': len(sales),
            'top_traded_tickers': [{'ticker': t, 'count': c} for t, c in top_tickers],
            'most_active_politicians': [{'name': n, 'count': c} for n, c in top_pols],
            'avg_days_to_disclose': round(avg_days, 1),
            'data_source': self._source_used or 'Unknown',
            'source_errors': self._source_errors,
            'newest_trade_date': newest_date,
            'oldest_trade_date': oldest_date,
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
