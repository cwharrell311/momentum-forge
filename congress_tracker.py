"""
Congressional Stock Trade Tracker
Fetches and processes stock trades reported by members of Congress.
Data sources:
  1. FMP API (primary, requires paid plan)
  2. Senate Stock Watcher GitHub data (free, Senate trades only)
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
import re
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Senator party/state lookup for enrichment
# Covers recent and current senators (2020-2025+)
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


class CongressTracker:
    """
    Tracks congressional stock trades from multiple data sources.
    """

    FMP_BASE_URL = "https://financialmodelingprep.com/api/v4"
    SENATE_WATCHER_URL = "https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/master/aggregate/all_transactions.json"

    def __init__(self, fmp_api_key: Optional[str] = None):
        self.fmp_api_key = fmp_api_key
        self._cache = None
        self._cache_time = None
        self._cache_ttl = 3600  # Cache for 1 hour

    def get_trades(self, days_back: int = 365) -> List[CongressTrade]:
        """
        Fetch recent congressional trades. Tries FMP first, then Senate Stock Watcher.
        """
        # Return cache if fresh
        if self._cache is not None and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < self._cache_ttl:
                logger.info(f"Returning {len(self._cache)} cached trades")
                return self._cache

        trades = []

        # Method 1: Try FMP API (requires paid plan)
        if self.fmp_api_key:
            trades = self._fetch_fmp_trades(days_back)

        # Method 2: Fallback to Senate Stock Watcher (free, always works)
        if not trades:
            logger.info("FMP congressional data unavailable, trying Senate Stock Watcher...")
            trades = self._fetch_senate_watcher(days_back)

        # Sort by trade date (newest first)
        trades.sort(key=lambda t: t.trade_date, reverse=True)

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

    def _fetch_senate_watcher(self, days_back: int) -> List[CongressTrade]:
        """
        Fetch trades from Senate Stock Watcher GitHub data.
        Source: https://github.com/timothycarambat/senate-stock-watcher-data
        Contains ~8000+ Senate stock trades with ticker, senator, date, type, amount.
        Note: This is a historical dataset. All records are loaded regardless of days_back.
        """
        trades = []

        try:
            logger.info("Fetching Senate Stock Watcher data from GitHub...")
            response = requests.get(self.SENATE_WATCHER_URL, timeout=60)
            logger.info(f"Senate Stock Watcher response: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"Senate Stock Watcher returned {response.status_code}")
                return trades

            data = response.json()
            logger.info(f"Senate Stock Watcher: {len(data)} total records")

            for item in data:
                try:
                    trade = self._parse_senate_watcher_trade(item)
                    if trade:
                        trades.append(trade)
                except Exception as e:
                    logger.debug(f"Error parsing Senate Watcher trade: {e}")
                    continue

            logger.info(f"Senate Stock Watcher: {len(trades)} stock trades loaded")

        except requests.exceptions.Timeout:
            logger.error("Senate Stock Watcher request timed out")
        except requests.exceptions.ConnectionError:
            logger.error("Senate Stock Watcher connection error")
        except Exception as e:
            logger.error(f"Error fetching Senate Stock Watcher data: {e}")

        return trades

    def _parse_senate_watcher_trade(self, item: Dict) -> Optional[CongressTrade]:
        """
        Parse a Senate Stock Watcher JSON record.

        Record format:
        {
            "transaction_date": "11/10/2020",
            "owner": "Spouse",
            "ticker": "BYND",
            "asset_description": "Beyond Meat, Inc.",
            "asset_type": "Stock",
            "type": "Sale (Full)",
            "amount": "$50,001 - $100,000",
            "comment": "--",
            "senator": "Ron L Wyden",
            "ptr_link": "https://efdsearch.senate.gov/..."
        }
        """
        ticker = item.get('ticker', '').strip()
        if not ticker or ticker == '--' or ticker == 'N/A' or ticker == '':
            return None

        # Skip non-stock assets
        asset_type = item.get('asset_type', '')
        if asset_type and asset_type.lower() not in ('stock', 'stock option', ''):
            return None

        senator = item.get('senator', 'Unknown').strip()

        # Look up party and state
        party, state = self._lookup_senator_info(senator)

        # Parse transaction date (MM/DD/YYYY format)
        raw_date = item.get('transaction_date', '').strip()
        trade_date = ''
        if raw_date:
            try:
                dt = datetime.strptime(raw_date, '%m/%d/%Y')
                trade_date = dt.strftime('%Y-%m-%d')
            except ValueError:
                trade_date = raw_date

        # Disclosure date: estimate from ptr_link or use trade_date
        # The dataset doesn't have a separate disclosure date field,
        # so we'll use the trade date as an approximation
        disclosure_date = trade_date

        amount_str = item.get('amount', '$0 - $0')
        amount_low, amount_high = self._parse_amount_range(amount_str)

        trade_type = self._normalize_trade_type(item.get('type', 'Unknown'))
        owner = item.get('owner', 'Self').strip()
        if owner == '--':
            owner = 'Self'

        return CongressTrade(
            politician=senator,
            party=party,
            chamber='Senate',
            state=state,
            ticker=ticker.upper(),
            asset_description=item.get('asset_description', ticker),
            trade_type=trade_type,
            trade_date=trade_date,
            disclosure_date=disclosure_date,
            amount_range=amount_str,
            amount_low=amount_low,
            amount_high=amount_high,
            days_to_disclose=0,  # Not available in this dataset
            owner=owner,
        )

    def _lookup_senator_info(self, senator_name: str) -> tuple:
        """Look up party and state for a senator by name."""
        # Direct lookup
        if senator_name in SENATOR_PARTIES:
            return SENATOR_PARTIES[senator_name]

        # Normalize: remove periods, commas, Jr/Sr/III suffixes, extra spaces
        normalized = re.sub(r'[.,]', '', senator_name)
        normalized = re.sub(r'\s+(Jr|Sr|III|II|IV)\s*$', '', normalized, flags=re.IGNORECASE)
        normalized = ' '.join(normalized.split())  # collapse whitespace

        # Try normalized direct lookup
        for known_name, info in SENATOR_PARTIES.items():
            known_normalized = re.sub(r'[.,]', '', known_name)
            known_normalized = ' '.join(known_normalized.split())
            if normalized.lower() == known_normalized.lower():
                return info

        # Try partial matching (last name + first initial)
        name_parts = normalized.split()
        if name_parts:
            # Get last real name part (skip Jr, etc.)
            last_name = name_parts[-1].lower()
            for known_name, (party, state) in SENATOR_PARTIES.items():
                known_parts = known_name.replace('.', '').split()
                known_last = known_parts[-1].lower()
                if last_name == known_last:
                    # Verify first initial matches
                    if len(name_parts) > 1 and len(known_parts) > 1:
                        if name_parts[0][0].lower() == known_parts[0][0].lower():
                            return (party, state)

        return ('Unknown', '')

    def _parse_amount_range(self, amount_str: str) -> tuple:
        """Parse amount range string like '$1,001 - $15,000' into (low, high)."""
        try:
            # Handle "Over $X" format
            over_match = re.search(r'Over\s+\$?([\d,]+)', str(amount_str), re.IGNORECASE)
            if over_match:
                val = float(over_match.group(1).replace(',', ''))
                return val, val * 2  # Estimate upper bound

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
