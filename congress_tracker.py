"""
Congressional Stock Trade Tracker
Fetches and processes stock trades reported by members of Congress.

Data source priority (tries in order, uses first that works):
  1. Capitol Trades API (free, current, Senate + House)
  2. Senate/House Stock Watcher websites (free, current)
  3. FMP API (requires paid plan)
  4. Senate Stock Watcher GitHub (historical fallback, 2012-2020 only)
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
import re
import json

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
    Tracks congressional stock trades from multiple free data sources.
    Tries live/current sources first, falls back to historical data.
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
        Fetch congressional trades. Tries sources in order of data freshness:
        1. Capitol Trades (current, both chambers)
        2. Senate/House Stock Watcher websites (current)
        3. FMP API (current, paid)
        4. Senate Stock Watcher GitHub (historical fallback)
        """
        # Return cache if fresh
        if self._cache is not None and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < self._cache_ttl:
                logger.info(f"Returning {len(self._cache)} cached trades")
                return self._cache

        trades = []
        source_used = None

        # Method 1: Capitol Trades (free, current data, both chambers)
        if not trades:
            trades = self._fetch_capitol_trades()
            if trades:
                source_used = "Capitol Trades"

        # Method 2: Senate/House Stock Watcher websites (free, current)
        if not trades:
            trades = self._fetch_stock_watcher_sites()
            if trades:
                source_used = "Stock Watcher Sites"

        # Method 3: FMP API (requires paid plan)
        if not trades and self.fmp_api_key:
            trades = self._fetch_fmp_trades(days_back)
            if trades:
                source_used = "FMP API"

        # Method 4: Historical GitHub data (last resort)
        if not trades:
            logger.warning("All live sources failed. Loading historical data as fallback...")
            trades = self._fetch_senate_watcher_github()
            if trades:
                source_used = "GitHub Historical (2012-2020)"

        # Sort by trade date (newest first)
        trades.sort(key=lambda t: t.trade_date, reverse=True)

        # Cache results
        self._cache = trades
        self._cache_time = datetime.now()

        logger.info(f"Source: {source_used} | Total trades: {len(trades)}")
        return trades

    # ------------------------------------------------------------------
    # Source 1: Capitol Trades (capitoltrades.com)
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
                return trades

            next_data = json.loads(match.group(1))
            # Navigate the Next.js data structure
            page_props = next_data.get('props', {}).get('pageProps', {})
            trade_data = page_props.get('trades', page_props.get('data', []))

            if isinstance(trade_data, dict):
                trade_data = trade_data.get('data', trade_data.get('trades', []))

            for item in trade_data:
                trade = self._parse_capitol_trade_item(item)
                if trade:
                    trades.append(trade)

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Failed to parse Capitol Trades HTML: {e}")

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
    # Source 2: Senate/House Stock Watcher websites
    # ------------------------------------------------------------------
    def _fetch_stock_watcher_sites(self) -> List[CongressTrade]:
        """
        Fetch from senatestockwatcher.com and housestockwatcher.com.
        These sites serve current data and have JSON API endpoints.
        """
        trades = []

        # Senate Stock Watcher
        senate_urls = [
            "https://senatestockwatcher.com/api/transactions",
            "https://senatestockwatcher.com/api/stats/allSenatorTransactions",
        ]
        for url in senate_urls:
            try:
                logger.info(f"Trying Senate Stock Watcher API: {url}")
                resp = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
                logger.info(f"Senate Stock Watcher API response: {resp.status_code}")
                if resp.status_code == 200:
                    data = resp.json()
                    items = data if isinstance(data, list) else data.get('data', data.get('transactions', []))
                    for item in items:
                        trade = self._parse_stock_watcher_trade(item, 'Senate')
                        if trade:
                            trades.append(trade)
                    if trades:
                        logger.info(f"Senate Stock Watcher API: {len(trades)} trades")
                        break
            except Exception as e:
                logger.info(f"Senate Stock Watcher API failed: {e}")

        # House Stock Watcher
        house_urls = [
            "https://housestockwatcher.com/api/transactions",
            "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json",
        ]
        house_trades = []
        for url in house_urls:
            try:
                logger.info(f"Trying House Stock Watcher: {url}")
                resp = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
                logger.info(f"House Stock Watcher response: {resp.status_code}")
                if resp.status_code == 200:
                    data = resp.json()
                    items = data if isinstance(data, list) else data.get('data', data.get('transactions', []))
                    for item in items:
                        trade = self._parse_stock_watcher_trade(item, 'House')
                        if trade:
                            house_trades.append(trade)
                    if house_trades:
                        logger.info(f"House Stock Watcher: {len(house_trades)} trades")
                        break
            except Exception as e:
                logger.info(f"House Stock Watcher failed: {e}")

        trades.extend(house_trades)
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
    # Source 3: FMP API (paid)
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
    # Source 4: Historical GitHub data (fallback)
    # ------------------------------------------------------------------
    def _fetch_senate_watcher_github(self) -> List[CongressTrade]:
        """
        Last resort: historical Senate trades from GitHub (2012-2020).
        Only used if all live sources fail.
        """
        trades = []
        try:
            logger.info("Fetching historical Senate data from GitHub...")
            response = requests.get(self.SENATE_WATCHER_URL, timeout=60)
            if response.status_code != 200:
                return trades

            data = response.json()
            logger.info(f"GitHub historical data: {len(data)} total records")

            for item in data:
                try:
                    trade = self._parse_senate_watcher_trade(item)
                    if trade:
                        trades.append(trade)
                except Exception:
                    continue

            logger.info(f"GitHub historical: {len(trades)} stock trades loaded (NOTE: data is from 2012-2020)")

        except Exception as e:
            logger.error(f"GitHub historical data error: {e}")

        return trades

    def _parse_senate_watcher_trade(self, item: Dict) -> Optional[CongressTrade]:
        """Parse a Senate Stock Watcher GitHub JSON record."""
        ticker = item.get('ticker', '').strip()
        if not ticker or ticker == '--' or ticker == 'N/A' or ticker == '':
            return None

        asset_type = item.get('asset_type', '')
        if asset_type and asset_type.lower() not in ('stock', 'stock option', ''):
            return None

        senator = item.get('senator', 'Unknown').strip()
        party, state = self._lookup_senator_info(senator)

        raw_date = item.get('transaction_date', '').strip()
        trade_date = self._standardize_date(raw_date)
        disclosure_date = trade_date

        amount_str = item.get('amount', '$0 - $0')
        amount_low, amount_high = self._parse_amount_range(amount_str)

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
            trade_type=self._normalize_trade_type(item.get('type', 'Unknown')),
            trade_date=trade_date,
            disclosure_date=disclosure_date,
            amount_range=amount_str,
            amount_low=amount_low,
            amount_high=amount_high,
            days_to_disclose=0,
            owner=owner,
        )

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
