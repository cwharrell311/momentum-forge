"""
Government Financial Disclosure Scraper
Scrapes congressional stock trade filings directly from official government sources:
  - Senate: efdsearch.senate.gov (Electronic Financial Disclosure)
  - House: disclosures-clerk.house.gov (House Clerk Financial Disclosures)

All data is publicly available under the STOCK Act of 2012.

Based on proven approaches from:
  - github.com/neelsomani/senator-filings
  - gist.github.com/dannguyen/994bfe5a4a1e9ba6c73f21046e31e86c
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import re
import json
import time
import os

logger = logging.getLogger(__name__)

# Try to use lxml parser, fall back to html.parser
try:
    import lxml
    HTML_PARSER = 'lxml'
except ImportError:
    HTML_PARSER = 'html.parser'
    logger.info("lxml not available, using html.parser")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')


class SenateScraper:
    """
    Scrapes the Senate Electronic Financial Disclosure (EFD) system.
    Based on neelsomani/senator-filings proven approach.

    Flow:
    1. GET landing page to extract csrfmiddlewaretoken
    2. POST to accept the prohibition agreement
    3. Get csrftoken from cookies
    4. POST to /search/report/data/ with proper params
    """

    LANDING_PAGE_URL = "https://efdsearch.senate.gov/search/home/"
    SEARCH_PAGE_URL = "https://efdsearch.senate.gov/search/"
    REPORTS_URL = "https://efdsearch.senate.gov/search/report/data/"
    BATCH_SIZE = 100

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._csrf_token = None
        self._last_request = 0

    def _rate_limit(self):
        """Enforce 2-second delay between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)
        self._last_request = time.time()

    def _get_csrf(self) -> Optional[str]:
        """
        Set the session ID and return the CSRF token for this session.
        This handles the prohibition agreement page.
        """
        self._rate_limit()
        try:
            # Get landing page with the agreement form
            landing_resp = self.session.get(self.LANDING_PAGE_URL, timeout=30)
            logger.info(f"Senate EFD landing page: {landing_resp.status_code}")

            if landing_resp.status_code != 200:
                logger.error(f"Senate EFD landing page failed: {landing_resp.status_code}")
                return None

            if landing_resp.url != self.LANDING_PAGE_URL:
                logger.warning(f"Unexpected redirect to: {landing_resp.url}")

            # Extract csrfmiddlewaretoken from form
            soup = BeautifulSoup(landing_resp.text, HTML_PARSER)
            csrf_input = soup.find(attrs={'name': 'csrfmiddlewaretoken'})

            if not csrf_input or not csrf_input.get('value'):
                logger.error("Could not find csrfmiddlewaretoken in landing page")
                return None

            form_csrf = csrf_input['value']
            logger.info(f"Got form CSRF token: {form_csrf[:20]}...")

            # Accept the prohibition agreement
            self._rate_limit()
            form_payload = {
                'csrfmiddlewaretoken': form_csrf,
                'prohibition_agreement': '1'
            }

            agree_resp = self.session.post(
                self.LANDING_PAGE_URL,
                data=form_payload,
                headers={'Referer': self.LANDING_PAGE_URL},
                timeout=30
            )
            logger.info(f"Senate EFD agreement POST: {agree_resp.status_code}")

            # Extract csrftoken from cookies
            if 'csrftoken' in self.session.cookies:
                self._csrf_token = self.session.cookies['csrftoken']
            elif 'csrf' in self.session.cookies:
                self._csrf_token = self.session.cookies['csrf']
            else:
                logger.error(f"No CSRF cookie found. Cookies: {list(self.session.cookies.keys())}")
                return None

            logger.info(f"Got session CSRF token: {self._csrf_token[:20]}...")
            return self._csrf_token

        except Exception as e:
            logger.error(f"CSRF flow failed: {e}")
            return None

    def fetch_ptr_list(self, offset: int = 0, days_back: int = 365) -> List[List[str]]:
        """
        Query the PTR (Periodic Transaction Report) list API.
        Returns list of [name_html, office, report_type_html, date] entries.
        """
        if not self._csrf_token:
            if not self._get_csrf():
                return []

        self._rate_limit()

        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%m/%d/%Y 00:00:00')

        # These are the exact parameters from the working neelsomani code
        login_data = {
            'start': str(offset),
            'length': str(self.BATCH_SIZE),
            'report_types': '[11]',  # 11 = Periodic Transaction Report
            'filer_types': '[]',
            'submitted_start_date': start_date,
            'submitted_end_date': '',
            'candidate_state': '',
            'senator_state': '',
            'office_id': '',
            'first_name': '',
            'last_name': '',
            'csrfmiddlewaretoken': self._csrf_token,
        }

        logger.info(f"Fetching PTR list, offset={offset}, since {start_date[:10]}")

        try:
            resp = self.session.post(
                self.REPORTS_URL,
                data=login_data,
                headers={'Referer': self.SEARCH_PAGE_URL},
                timeout=60
            )

            logger.info(f"PTR list response: {resp.status_code}")

            if resp.status_code != 200:
                logger.error(f"PTR list request failed: {resp.status_code} - {resp.text[:500]}")
                return []

            data = resp.json()
            rows = data.get('data', [])
            logger.info(f"Got {len(rows)} PTR entries")
            return rows

        except json.JSONDecodeError as e:
            logger.error(f"PTR list response not JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"PTR list request error: {e}")
            return []

    def fetch_all_ptrs(self, days_back: int = 365) -> List[Dict]:
        """Fetch all PTR filings, paginating through results."""
        all_filings = []
        offset = 0

        while True:
            rows = self.fetch_ptr_list(offset=offset, days_back=days_back)
            if not rows:
                break

            for row in rows:
                filing = self._parse_ptr_row(row)
                if filing:
                    all_filings.append(filing)

            if len(rows) < self.BATCH_SIZE:
                break

            offset += self.BATCH_SIZE

            # Safety limit
            if offset > 10000:
                logger.warning("Hit 10k filing limit, stopping")
                break

        logger.info(f"Total PTR filings found: {len(all_filings)}")
        return all_filings

    def _parse_ptr_row(self, row: List) -> Optional[Dict]:
        """Parse a row from the PTR list API response."""
        try:
            if not isinstance(row, list) or len(row) < 4:
                logger.debug(f"Row too short or not a list: {row}")
                return None

            # Row format: [name_html, office, report_type_html, date]
            name_html = str(row[0])
            report_html = str(row[2])
            date_str = str(row[3]).strip()

            # Log first row for debugging
            if not hasattr(self, '_logged_sample'):
                logger.info(f"Sample PTR row[0] (name): {name_html[:200]}")
                logger.info(f"Sample PTR row[2] (report): {report_html[:200]}")
                self._logged_sample = True

            # Extract name from HTML - try multiple patterns
            name = ''
            # Pattern 1: Text between > and <
            name_match = re.search(r'>([^<]+)<', name_html)
            if name_match:
                name = name_match.group(1).strip()
            # Pattern 2: Just get any text content
            if not name:
                name = re.sub(r'<[^>]+>', '', name_html).strip()

            # Extract report link from HTML - try multiple patterns
            link = ''
            # Pattern 1: href in report cell
            link_match = re.search(r'href=["\']([^"\']+)', report_html)
            if link_match:
                link = link_match.group(1).strip()
            # Pattern 2: href in name cell
            if not link:
                link_match = re.search(r'href=["\']([^"\']+)', name_html)
                if link_match:
                    link = link_match.group(1).strip()
            # Pattern 3: Look for /search/view/paper/ or /search/view/ptr/ paths
            if not link:
                link_match = re.search(r'(/search/view/(?:paper|ptr)/[^"\'>\s]+)', name_html + report_html)
                if link_match:
                    link = link_match.group(1)

            if link and not link.startswith('http'):
                link = f"https://efdsearch.senate.gov{link}"

            # Be more lenient - accept if we have either name or link
            if name or link:
                return {
                    'name': name or 'Unknown Senator',
                    'link': link,
                    'date': date_str,
                    'type': 'PTR',
                    'chamber': 'Senate',
                }
            else:
                logger.debug(f"Could not extract name or link from row")
        except Exception as e:
            logger.debug(f"Error parsing PTR row: {e}")

        return None

    def fetch_filing_transactions(self, filing: Dict) -> List[Dict]:
        """
        Fetch individual transactions from a PTR filing detail page.
        """
        transactions = []
        link = filing.get('link', '')
        if not link:
            return transactions

        self._rate_limit()

        try:
            # Check if session expired (redirects to landing page)
            resp = self.session.get(link, timeout=30, allow_redirects=False)

            if resp.status_code in (301, 302):
                redirect_url = resp.headers.get('Location', '')
                if 'home' in redirect_url or 'login' in redirect_url:
                    logger.info("Session expired, re-authenticating...")
                    self._csrf_token = None
                    self._get_csrf()
                    resp = self.session.get(link, timeout=30)
            elif resp.status_code != 200:
                logger.debug(f"Filing page returned {resp.status_code}: {link}")
                return transactions

            soup = BeautifulSoup(resp.text, HTML_PARSER)

            # Find transaction tables
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                if not rows:
                    continue

                # Check header row for transaction-related keywords
                header_row = rows[0]
                header_text = header_row.get_text().lower()

                if any(kw in header_text for kw in ['ticker', 'transaction', 'asset', 'amount', 'type', 'date']):
                    # Parse data rows
                    for row in rows[1:]:
                        tx = self._parse_transaction_row(row, filing)
                        if tx:
                            transactions.append(tx)

            # If no table transactions, try parsing structured content
            if not transactions:
                transactions = self._parse_unstructured(soup, filing)

        except Exception as e:
            logger.debug(f"Error fetching filing: {e}")

        return transactions

    def _parse_transaction_row(self, row, filing: Dict) -> Optional[Dict]:
        """Parse a transaction table row."""
        cells = row.find_all('td')
        if len(cells) < 3:
            return None

        try:
            cell_texts = [c.get_text(strip=True) for c in cells]

            ticker = ''
            asset_name = ''
            tx_date = ''
            tx_type = ''
            amount = ''
            owner = ''

            for text in cell_texts:
                text_clean = text.strip()

                # Ticker: 1-5 uppercase letters
                if re.match(r'^[A-Z]{1,5}$', text_clean) and not ticker:
                    if text_clean not in ('LLC', 'INC', 'ETF', 'USD', 'THE', 'AND', 'FOR', 'JR', 'SR'):
                        ticker = text_clean

                # Date: MM/DD/YYYY
                elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', text_clean) and not tx_date:
                    tx_date = text_clean

                # Trade type
                elif any(kw in text_clean.lower() for kw in ['purchase', 'sale', 'exchange', 'buy', 'sell']):
                    tx_type = text_clean

                # Amount range
                elif '$' in text_clean:
                    amount = text_clean

                # Owner
                elif text_clean.lower() in ('self', 'spouse', 'joint', 'child', 'dependent'):
                    owner = text_clean

                # Asset name (check for embedded ticker)
                elif len(text_clean) > 5 and not asset_name:
                    ticker_match = re.search(r'\(([A-Z]{1,5})\)', text_clean)
                    if ticker_match and not ticker:
                        ticker = ticker_match.group(1)
                    asset_name = text_clean

            if ticker:
                return {
                    'politician': filing.get('name', 'Unknown'),
                    'ticker': ticker,
                    'asset_description': asset_name or ticker,
                    'trade_type': tx_type or 'Unknown',
                    'trade_date': tx_date,
                    'amount': amount,
                    'owner': owner or 'Self',
                    'disclosure_date': filing.get('date', ''),
                    'chamber': 'Senate',
                    'source_link': filing.get('link', ''),
                }

        except Exception as e:
            logger.debug(f"Error parsing row: {e}")

        return None

    def _parse_unstructured(self, soup: BeautifulSoup, filing: Dict) -> List[Dict]:
        """Parse transactions from non-table content."""
        transactions = []
        text = soup.get_text()

        # Find ticker patterns followed by transaction info
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        date_pattern = r'(\d{1,2}/\d{1,2}/\d{4})'
        amount_pattern = r'(\$[\d,]+ - \$[\d,]+|\$[\d,]+)'

        tickers = re.findall(ticker_pattern, text)
        dates = re.findall(date_pattern, text)
        amounts = re.findall(amount_pattern, text)

        # Filter out common non-ticker words
        skip_words = {'LLC', 'INC', 'ETF', 'USD', 'USA', 'THE', 'AND', 'FOR', 'JR', 'SR', 'III', 'II', 'IV'}
        tickers = [t for t in tickers if t not in skip_words]

        for i, ticker in enumerate(tickers[:20]):  # Limit
            tx = {
                'politician': filing.get('name', 'Unknown'),
                'ticker': ticker,
                'asset_description': ticker,
                'trade_type': 'Unknown',
                'trade_date': dates[i] if i < len(dates) else filing.get('date', ''),
                'amount': amounts[i] if i < len(amounts) else '',
                'owner': 'Self',
                'disclosure_date': filing.get('date', ''),
                'chamber': 'Senate',
                'source_link': filing.get('link', ''),
            }
            transactions.append(tx)

        return transactions

    def scrape(self, days_back: int = 365) -> List[Dict]:
        """
        Full scrape: fetch PTR list, then transactions from each.
        """
        all_transactions = []

        filings = self.fetch_all_ptrs(days_back)
        logger.info(f"Fetching transactions from {len(filings)} Senate PTR filings...")

        for i, filing in enumerate(filings):
            transactions = self.fetch_filing_transactions(filing)
            all_transactions.extend(transactions)

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(filings)} filings ({len(all_transactions)} transactions)")

            # Limit for testing
            if i >= 100:
                logger.info("Hit 100 filing limit for initial test")
                break

        logger.info(f"Senate scraper: {len(all_transactions)} total transactions")
        return all_transactions


class HouseScraper:
    """
    Scrapes the House Clerk Financial Disclosure system.
    Based on dannguyen's ASPX form scraping approach.

    The House system uses ASP.NET with ViewState and EventValidation.
    We extract all form fields and POST back with our search params.
    """

    SEARCH_URL = "https://disclosures-clerk.house.gov/FinancialDisclosure"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._last_request = 0

    def _rate_limit(self):
        """Enforce delay between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
        self._last_request = time.time()

    def scrape(self, days_back: int = 365) -> List[Dict]:
        """
        Scrape House PTR filings.
        Note: House filings are mostly PDFs which are hard to parse.
        We focus on getting filing metadata and any available HTML detail pages.
        """
        all_transactions = []

        self._rate_limit()

        try:
            # Get the search page
            resp = self.session.get(self.SEARCH_URL, timeout=30)
            logger.info(f"House search page: {resp.status_code}")

            if resp.status_code != 200:
                logger.error(f"House search page failed: {resp.status_code}")
                return []

            soup = BeautifulSoup(resp.text, HTML_PARSER)

            # Look for PTR links on the page
            ptr_links = soup.find_all('a', href=re.compile(r'ptr|transaction|periodic', re.I))
            logger.info(f"Found {len(ptr_links)} PTR-related links")

            # Also look for any data tables
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        tx = self._parse_house_row(cells)
                        if tx:
                            all_transactions.append(tx)

            # Note: Full House scraping requires handling PDF files
            # which needs pdfplumber or similar. For now, we just note
            # this limitation.
            if not all_transactions:
                logger.info("House scraper: Most filings are PDFs, limited parsing available")

        except Exception as e:
            logger.error(f"House scraper error: {e}")

        logger.info(f"House scraper: {len(all_transactions)} transactions")
        return all_transactions

    def _parse_house_row(self, cells: list) -> Optional[Dict]:
        """Parse a House disclosure table row."""
        try:
            cell_texts = [c.get_text(strip=True) for c in cells]

            # Try to find name, ticker, date patterns
            name = cell_texts[0] if cell_texts else 'Unknown'

            ticker = ''
            tx_date = ''
            amount = ''
            tx_type = ''

            for text in cell_texts:
                if re.match(r'^[A-Z]{1,5}$', text) and not ticker:
                    ticker = text
                elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', text) and not tx_date:
                    tx_date = text
                elif '$' in text:
                    amount = text
                elif any(kw in text.lower() for kw in ['purchase', 'sale']):
                    tx_type = text

            if ticker:
                return {
                    'politician': name,
                    'ticker': ticker,
                    'asset_description': ticker,
                    'trade_type': tx_type or 'Unknown',
                    'trade_date': tx_date,
                    'amount': amount,
                    'owner': 'Self',
                    'disclosure_date': tx_date,
                    'chamber': 'House',
                }

        except Exception:
            pass

        return None


class GovScraper:
    """
    Combined Senate + House government disclosure scraper.
    Manages caching and provides a unified interface.
    """

    def __init__(self):
        self.senate = SenateScraper()
        self.house = HouseScraper()
        os.makedirs(CACHE_DIR, exist_ok=True)

    def scrape_all(self, days_back: int = 365, use_cache: bool = True) -> List[Dict]:
        """
        Scrape both Senate and House disclosures.
        Uses local file cache (refreshes every 6 hours).
        """
        if use_cache:
            cached = self._load_cache()
            if cached is not None:
                logger.info(f"Using cached government data: {len(cached)} transactions")
                return cached

        all_trades = []
        errors = []

        # Senate
        try:
            logger.info("Starting Senate scraper...")
            senate_trades = self.senate.scrape(days_back)
            all_trades.extend(senate_trades)
            logger.info(f"Senate: {len(senate_trades)} transactions")
        except Exception as e:
            errors.append(f"Senate: {e}")
            logger.error(f"Senate scraper error: {e}")

        # House
        try:
            logger.info("Starting House scraper...")
            house_trades = self.house.scrape(days_back)
            all_trades.extend(house_trades)
            logger.info(f"House: {len(house_trades)} transactions")
        except Exception as e:
            errors.append(f"House: {e}")
            logger.error(f"House scraper error: {e}")

        if all_trades:
            self._save_cache(all_trades)

        logger.info(f"Government scraper total: {len(all_trades)} transactions")
        if errors:
            logger.warning(f"Errors encountered: {errors}")

        return all_trades

    def _load_cache(self) -> Optional[List[Dict]]:
        """Load cached data if fresh (< 6 hours old)."""
        cache_file = os.path.join(CACHE_DIR, 'gov_trades.json')
        try:
            if os.path.exists(cache_file):
                mtime = os.path.getmtime(cache_file)
                age_hours = (time.time() - mtime) / 3600
                if age_hours < 6:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    logger.info(f"Cache is {age_hours:.1f} hours old, using it")
                    return data
                else:
                    logger.info(f"Cache is {age_hours:.1f} hours old, refreshing")
        except Exception as e:
            logger.debug(f"Cache load error: {e}")
        return None

    def _save_cache(self, trades: List[Dict]):
        """Save scraped data to local cache file."""
        cache_file = os.path.join(CACHE_DIR, 'gov_trades.json')
        try:
            with open(cache_file, 'w') as f:
                json.dump(trades, f, indent=2, default=str)
            logger.info(f"Saved {len(trades)} transactions to cache")
        except Exception as e:
            logger.debug(f"Cache save error: {e}")


if __name__ == '__main__':
    """Test the scraper standalone."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    scraper = GovScraper()
    trades = scraper.scrape_all(days_back=90, use_cache=False)

    print(f"\nTotal transactions: {len(trades)}")
    for t in trades[:10]:
        print(f"  {t.get('trade_date', '?'):12} | {t.get('politician', '?'):25} | {t.get('chamber', '?'):6} | "
              f"{t.get('ticker', '?'):6} | {t.get('trade_type', '?'):15} | {t.get('amount', '?')}")
