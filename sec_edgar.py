"""
SEC EDGAR API Integration
Direct access to company financials from SEC filings (10-Q, 10-K).
100% free, no API key required.

SEC API Documentation: https://www.sec.gov/edgar/sec-api-documentation
"""

import requests
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

class SECEdgar:
    """
    SEC EDGAR API client for fetching company financials.

    Uses the SEC's free APIs:
    - Company Facts API: Structured XBRL financial data
    - Submissions API: Filing history and company info
    """

    BASE_URL = "https://data.sec.gov"

    # SEC requires a User-Agent with contact info
    HEADERS = {
        "User-Agent": "MomentumForge/1.0 (contact@example.com)",
        "Accept-Encoding": "gzip, deflate"
    }

    # Cache for ticker -> CIK mapping
    _ticker_to_cik: Dict[str, str] = {}
    _cik_cache_loaded = False

    # Cache directory
    CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache', 'sec')

    def __init__(self, cache_hours: int = 24):
        """
        Initialize SEC EDGAR client.

        Args:
            cache_hours: How long to cache SEC data (default 24 hours)
        """
        self.cache_hours = cache_hours
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._load_ticker_mapping()

    def _load_ticker_mapping(self):
        """Load the SEC's ticker -> CIK mapping file."""
        if self._cik_cache_loaded:
            return

        try:
            # SEC provides a mapping file (note: www.sec.gov, not data.sec.gov)
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=self.HEADERS, timeout=30)

            if response.status_code == 200:
                data = response.json()
                # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
                for entry in data.values():
                    ticker = entry.get('ticker', '').upper()
                    cik = str(entry.get('cik_str', '')).zfill(10)  # Pad to 10 digits
                    if ticker and cik:
                        self._ticker_to_cik[ticker] = cik

                logger.info(f"Loaded {len(self._ticker_to_cik)} ticker->CIK mappings from SEC")
                self._cik_cache_loaded = True
            else:
                logger.warning(f"Failed to load SEC ticker mapping: {response.status_code}")

        except Exception as e:
            logger.error(f"Error loading SEC ticker mapping: {e}")

    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Get the SEC CIK number for a ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            10-digit CIK string or None if not found
        """
        self._load_ticker_mapping()
        return self._ticker_to_cik.get(ticker.upper())

    def _get_cached_data(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache if fresh."""
        cache_file = os.path.join(self.CACHE_DIR, f"{cache_key}.json")

        try:
            if os.path.exists(cache_file):
                mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - mtime < timedelta(hours=self.cache_hours):
                    with open(cache_file, 'r') as f:
                        return json.load(f)
        except Exception:
            pass

        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save data to cache."""
        cache_file = os.path.join(self.CACHE_DIR, f"{cache_key}.json")

        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to cache SEC data: {e}")

    def get_company_facts(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch structured financial data from SEC Company Facts API.

        This is the gold mine - contains all XBRL-tagged financial data from
        10-K and 10-Q filings in a structured format.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Company facts dictionary or None
        """
        cik = self.get_cik(ticker)
        if not cik:
            logger.warning(f"No CIK found for ticker {ticker}")
            return None

        # Check cache first
        cache_key = f"facts_{cik}"
        cached = self._get_cached_data(cache_key)
        if cached:
            logger.debug(f"Using cached SEC data for {ticker}")
            return cached

        try:
            url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"

            # SEC rate limit: 10 requests per second
            time.sleep(0.1)

            response = requests.get(url, headers=self.HEADERS, timeout=30)

            if response.status_code == 200:
                data = response.json()
                self._save_to_cache(cache_key, data)
                logger.info(f"Fetched SEC company facts for {ticker}")
                return data
            else:
                logger.warning(f"SEC API error for {ticker}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error fetching SEC data for {ticker}: {e}")
            return None

    def get_quarterly_revenue(self, ticker: str, num_quarters: int = 8) -> List[Dict[str, Any]]:
        """
        Get quarterly revenue figures from SEC filings.

        Args:
            ticker: Stock ticker symbol
            num_quarters: Number of quarters to return

        Returns:
            List of dicts with 'period', 'value', 'filed' keys
        """
        facts = self.get_company_facts(ticker)
        if not facts:
            return []

        revenues = []

        # Try different revenue concept names (companies use different tags)
        revenue_concepts = [
            ('us-gaap', 'Revenues'),
            ('us-gaap', 'RevenueFromContractWithCustomerExcludingAssessedTax'),
            ('us-gaap', 'SalesRevenueNet'),
            ('us-gaap', 'SalesRevenueGoodsNet'),
            ('us-gaap', 'SalesRevenueServicesNet'),
        ]

        for namespace, concept in revenue_concepts:
            try:
                concept_data = facts.get('facts', {}).get(namespace, {}).get(concept, {})
                units = concept_data.get('units', {})

                # Revenue is usually in USD
                usd_data = units.get('USD', [])

                if usd_data:
                    # Filter for quarterly data (10-Q filings, not annual)
                    quarterly = [
                        {
                            'period': item.get('end'),
                            'value': item.get('val'),
                            'filed': item.get('filed'),
                            'form': item.get('form'),
                            'frame': item.get('frame', '')
                        }
                        for item in usd_data
                        if item.get('form') in ['10-Q', '10-K'] and item.get('frame', '').startswith('CY')
                    ]

                    # Sort by period (most recent first) and dedupe
                    seen = set()
                    unique = []
                    for item in sorted(quarterly, key=lambda x: x['period'], reverse=True):
                        if item['period'] not in seen:
                            seen.add(item['period'])
                            unique.append(item)

                    if unique:
                        revenues = unique[:num_quarters]
                        break

            except Exception as e:
                logger.debug(f"Error parsing {concept} for {ticker}: {e}")
                continue

        return revenues

    def get_quarterly_eps(self, ticker: str, num_quarters: int = 8) -> List[Dict[str, Any]]:
        """
        Get quarterly EPS figures from SEC filings.

        Args:
            ticker: Stock ticker symbol
            num_quarters: Number of quarters to return

        Returns:
            List of dicts with 'period', 'value', 'filed' keys
        """
        facts = self.get_company_facts(ticker)
        if not facts:
            return []

        eps_data = []

        # Try different EPS concept names
        eps_concepts = [
            ('us-gaap', 'EarningsPerShareBasic'),
            ('us-gaap', 'EarningsPerShareDiluted'),
            ('us-gaap', 'BasicEarningsLossPerShare'),
        ]

        for namespace, concept in eps_concepts:
            try:
                concept_data = facts.get('facts', {}).get(namespace, {}).get(concept, {})
                units = concept_data.get('units', {})

                # EPS is usually in USD/shares
                usd_data = units.get('USD/shares', [])

                if usd_data:
                    # Filter for quarterly data
                    quarterly = [
                        {
                            'period': item.get('end'),
                            'value': item.get('val'),
                            'filed': item.get('filed'),
                            'form': item.get('form'),
                            'frame': item.get('frame', '')
                        }
                        for item in usd_data
                        if item.get('form') in ['10-Q', '10-K'] and item.get('frame', '').startswith('CY')
                    ]

                    # Sort by period and dedupe
                    seen = set()
                    unique = []
                    for item in sorted(quarterly, key=lambda x: x['period'], reverse=True):
                        if item['period'] not in seen:
                            seen.add(item['period'])
                            unique.append(item)

                    if unique:
                        eps_data = unique[:num_quarters]
                        break

            except Exception as e:
                logger.debug(f"Error parsing {concept} for {ticker}: {e}")
                continue

        return eps_data

    def get_quarterly_net_income(self, ticker: str, num_quarters: int = 8) -> List[Dict[str, Any]]:
        """
        Get quarterly net income figures from SEC filings.
        """
        facts = self.get_company_facts(ticker)
        if not facts:
            return []

        income_data = []

        income_concepts = [
            ('us-gaap', 'NetIncomeLoss'),
            ('us-gaap', 'ProfitLoss'),
            ('us-gaap', 'NetIncomeLossAvailableToCommonStockholdersBasic'),
        ]

        for namespace, concept in income_concepts:
            try:
                concept_data = facts.get('facts', {}).get(namespace, {}).get(concept, {})
                units = concept_data.get('units', {})
                usd_data = units.get('USD', [])

                if usd_data:
                    quarterly = [
                        {
                            'period': item.get('end'),
                            'value': item.get('val'),
                            'filed': item.get('filed'),
                            'form': item.get('form'),
                            'frame': item.get('frame', '')
                        }
                        for item in usd_data
                        if item.get('form') in ['10-Q', '10-K'] and item.get('frame', '').startswith('CY')
                    ]

                    seen = set()
                    unique = []
                    for item in sorted(quarterly, key=lambda x: x['period'], reverse=True):
                        if item['period'] not in seen:
                            seen.add(item['period'])
                            unique.append(item)

                    if unique:
                        income_data = unique[:num_quarters]
                        break

            except Exception as e:
                logger.debug(f"Error parsing {concept} for {ticker}: {e}")
                continue

        return income_data

    def get_financials_summary(self, ticker: str) -> Dict[str, Any]:
        """
        Get a summary of key financial metrics for a stock.

        Returns:
            Dict with revenue_growth_yoy, revenue_accelerating, eps_data, etc.
        """
        result = {
            'ticker': ticker,
            'revenue_growth_yoy': None,
            'revenue_accelerating': False,
            'latest_revenue': None,
            'latest_eps': None,
            'latest_net_income': None,
            'data_source': 'SEC EDGAR'
        }

        # Get quarterly revenue
        revenues = self.get_quarterly_revenue(ticker)
        if len(revenues) >= 5:
            current = revenues[0]['value']
            year_ago = revenues[4]['value']  # 4 quarters back

            if year_ago and year_ago > 0:
                result['revenue_growth_yoy'] = round(((current - year_ago) / year_ago) * 100, 1)
                result['latest_revenue'] = current

                # Check for acceleration
                if len(revenues) >= 6:
                    prev = revenues[1]['value']
                    prev_year_ago = revenues[5]['value']
                    if prev_year_ago and prev_year_ago > 0:
                        prev_growth = ((prev - prev_year_ago) / prev_year_ago) * 100
                        result['revenue_accelerating'] = result['revenue_growth_yoy'] > prev_growth

        # Get EPS
        eps = self.get_quarterly_eps(ticker)
        if eps:
            result['latest_eps'] = eps[0]['value']

        # Get net income
        net_income = self.get_quarterly_net_income(ticker)
        if net_income:
            result['latest_net_income'] = net_income[0]['value']

        return result


# Convenience function
def get_sec_financials(ticker: str) -> Dict[str, Any]:
    """Quick access to SEC financials for a ticker."""
    edgar = SECEdgar()
    return edgar.get_financials_summary(ticker)


if __name__ == "__main__":
    # Test the SEC EDGAR integration
    logging.basicConfig(level=logging.INFO)

    edgar = SECEdgar()

    # Test with Apple
    print("\n=== Testing SEC EDGAR for AAPL ===")

    cik = edgar.get_cik('AAPL')
    print(f"CIK: {cik}")

    revenues = edgar.get_quarterly_revenue('AAPL')
    print(f"\nQuarterly Revenue (last 4):")
    for r in revenues[:4]:
        print(f"  {r['period']}: ${r['value']:,.0f}")

    eps = edgar.get_quarterly_eps('AAPL')
    print(f"\nQuarterly EPS (last 4):")
    for e in eps[:4]:
        print(f"  {e['period']}: ${e['value']:.2f}")

    summary = edgar.get_financials_summary('AAPL')
    print(f"\nSummary:")
    print(f"  Revenue Growth YoY: {summary['revenue_growth_yoy']}%")
    print(f"  Revenue Accelerating: {summary['revenue_accelerating']}")
    print(f"  Latest EPS: ${summary['latest_eps']:.2f}")
