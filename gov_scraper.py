"""
Government Financial Disclosure Scraper
Scrapes congressional stock trade filings directly from official government sources:
  - Senate: efdsearch.senate.gov (Electronic Financial Disclosure)
  - House: disclosures-clerk.house.gov (House Clerk Financial Disclosures)

All data is publicly available under the STOCK Act of 2012.
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

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Local cache file path (next to this script)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
SENATE_CACHE = os.path.join(CACHE_DIR, 'senate_trades.json')
HOUSE_CACHE = os.path.join(CACHE_DIR, 'house_trades.json')


class SenateScraper:
    """
    Scrapes the Senate Electronic Financial Disclosure (EFD) system.
    URL: https://efdsearch.senate.gov/search/

    The Senate EFD provides Periodic Transaction Reports (PTRs) that
    contain individual stock trades by senators.
    """

    BASE_URL = "https://efdsearch.senate.gov"
    SEARCH_URL = f"{BASE_URL}/search/"
    REPORT_URL = f"{BASE_URL}/search/report/data/"
    AGREEMENT_URL = f"{BASE_URL}/search/home/confirm/"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._agreed = False

    def _accept_agreement(self) -> bool:
        """Accept the EFD search agreement/disclaimer page."""
        if self._agreed:
            return True

        try:
            # First, visit the home page to get cookies
            resp = self.session.get(f"{self.BASE_URL}/search/home/", timeout=15)
            logger.info(f"Senate EFD home page: {resp.status_code}")

            if resp.status_code != 200:
                return False

            # Check if there's an agreement page
            if 'agree' in resp.text.lower() or 'confirm' in resp.text.lower():
                # Extract CSRF token if present
                csrf = self._extract_csrf(resp.text)

                # POST to accept the agreement
                data = {'prohibition_agreement': '1'}
                if csrf:
                    data['csrfmiddlewaretoken'] = csrf
                    self.session.headers['X-CSRFToken'] = csrf

                agree_resp = self.session.post(
                    self.AGREEMENT_URL,
                    data=data,
                    headers={'Referer': f"{self.BASE_URL}/search/home/"},
                    timeout=15,
                    allow_redirects=True,
                )
                logger.info(f"Senate EFD agreement: {agree_resp.status_code}")

            self._agreed = True
            return True

        except Exception as e:
            logger.error(f"Senate EFD agreement failed: {e}")
            return False

    def _extract_csrf(self, html: str) -> Optional[str]:
        """Extract CSRF token from HTML page."""
        # Look for csrfmiddlewaretoken in form
        match = re.search(r'name=["\']csrfmiddlewaretoken["\'].*?value=["\']([^"\']+)', html)
        if match:
            return match.group(1)
        # Look for csrf token in cookie-style meta tag
        match = re.search(r'csrf[_-]?token["\']?\s*(?:content|value)=["\']([^"\']+)', html, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def fetch_recent_filings(self, days_back: int = 120) -> List[Dict]:
        """
        Search for recent Periodic Transaction Reports (PTR filings).
        Returns a list of filing metadata (senator name, date, link).
        """
        if not self._accept_agreement():
            logger.warning("Could not accept Senate EFD agreement")
            return []

        filings = []
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%m/%d/%Y')

        try:
            # The Senate EFD search uses AJAX POST requests
            search_data = {
                'start_date': start_date,
                'end_date': datetime.now().strftime('%m/%d/%Y'),
                'filer_type': '',  # All filers
                'report_type': '11',  # Periodic Transaction Report
                'submitted_start_date': start_date,
                'submitted_end_date': '',
                'candidate_state': '',
                'senator_state': '',
                'office_id': '',
            }

            csrf = self.session.cookies.get('csrftoken', '')
            if csrf:
                search_data['csrfmiddlewaretoken'] = csrf

            logger.info(f"Searching Senate EFD for PTR filings since {start_date}...")
            resp = self.session.post(
                self.REPORT_URL,
                data=search_data,
                headers={
                    'Referer': self.SEARCH_URL,
                    'X-Requested-With': 'XMLHttpRequest',
                },
                timeout=30,
            )
            logger.info(f"Senate EFD search response: {resp.status_code}")

            if resp.status_code != 200:
                # Try alternative: direct search page
                resp = self.session.get(
                    f"{self.SEARCH_URL}?report_type=11&submitted_start_date={start_date}",
                    timeout=30,
                )
                logger.info(f"Senate EFD alternative search: {resp.status_code}")

            if resp.status_code == 200:
                filings = self._parse_search_results(resp.text)
                logger.info(f"Found {len(filings)} Senate PTR filings")

        except Exception as e:
            logger.error(f"Senate EFD search failed: {e}")

        return filings

    def _parse_search_results(self, html: str) -> List[Dict]:
        """Parse the search results page for filing links."""
        filings = []
        soup = BeautifulSoup(html, 'html.parser')

        # Try parsing as JSON (AJAX response)
        try:
            data = json.loads(html)
            if isinstance(data, dict) and 'data' in data:
                for row in data['data']:
                    filing = self._parse_json_row(row)
                    if filing:
                        filings.append(filing)
                return filings
        except (json.JSONDecodeError, ValueError):
            pass

        # Parse as HTML table
        table = soup.find('table', {'id': 'filedReports'}) or soup.find('table')
        if not table:
            # Try tbody directly
            rows = soup.find_all('tr')
        else:
            rows = table.find_all('tr')

        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 4:
                filing = self._parse_html_row(cells)
                if filing:
                    filings.append(filing)

        return filings

    def _parse_json_row(self, row: Any) -> Optional[Dict]:
        """Parse a JSON row from AJAX response."""
        try:
            if isinstance(row, list):
                # Format: [name_html, office, report_type_html, date]
                name_match = re.search(r'>([^<]+)<', str(row[0]))
                link_match = re.search(r'href=["\']([^"\']+)', str(row[0]))
                name = name_match.group(1).strip() if name_match else ''
                link = link_match.group(1).strip() if link_match else ''

                # Report type link
                report_link_match = re.search(r'href=["\']([^"\']+)', str(row[2]))
                report_link = report_link_match.group(1).strip() if report_link_match else link

                date_str = str(row[3]).strip() if len(row) > 3 else ''

                if name and report_link:
                    if not report_link.startswith('http'):
                        report_link = f"{self.BASE_URL}{report_link}"
                    return {
                        'name': name,
                        'link': report_link,
                        'date': date_str,
                        'type': 'PTR',
                    }
        except Exception as e:
            logger.debug(f"Error parsing JSON row: {e}")
        return None

    def _parse_html_row(self, cells: list) -> Optional[Dict]:
        """Parse an HTML table row."""
        try:
            # First cell: name with link
            name_cell = cells[0]
            name = name_cell.get_text(strip=True)
            link_tag = name_cell.find('a') or (cells[2].find('a') if len(cells) > 2 else None)
            link = link_tag['href'] if link_tag and 'href' in link_tag.attrs else ''

            if not link.startswith('http') and link:
                link = f"{self.BASE_URL}{link}"

            # Date cell (usually last or second-to-last)
            date_str = cells[-1].get_text(strip=True) if cells else ''

            if name and link:
                return {
                    'name': name,
                    'link': link,
                    'date': date_str,
                    'type': 'PTR',
                }
        except Exception:
            pass
        return None

    def fetch_filing_transactions(self, filing: Dict) -> List[Dict]:
        """
        Fetch individual transactions from a PTR filing detail page.
        Each transaction is a stock trade (buy/sell) with ticker, amount, date.
        """
        transactions = []
        link = filing.get('link', '')
        if not link:
            return transactions

        try:
            time.sleep(0.5)  # Be respectful
            resp = self.session.get(link, timeout=15)

            if resp.status_code != 200:
                logger.debug(f"Filing detail page returned {resp.status_code}: {link}")
                return transactions

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Find transaction tables
            tables = soup.find_all('table')
            for table in tables:
                header_text = ''
                thead = table.find('thead') or table.find('tr')
                if thead:
                    header_text = thead.get_text().lower()

                # Look for transaction tables (contain columns like ticker, date, type, amount)
                if any(kw in header_text for kw in ['ticker', 'transaction', 'asset', 'amount', 'type']):
                    rows = table.find_all('tr')[1:]  # Skip header
                    for row in rows:
                        tx = self._parse_transaction_row(row, filing)
                        if tx:
                            transactions.append(tx)

            # If no table found, try parsing structured divs
            if not transactions:
                transactions = self._parse_structured_content(soup, filing)

        except Exception as e:
            logger.debug(f"Error fetching filing transactions: {e}")

        return transactions

    def _parse_transaction_row(self, row, filing: Dict) -> Optional[Dict]:
        """Parse a single transaction row from a filing detail page."""
        cells = row.find_all('td')
        if len(cells) < 4:
            return None

        try:
            cell_texts = [c.get_text(strip=True) for c in cells]

            # Try to identify columns by content patterns
            ticker = ''
            asset_name = ''
            tx_date = ''
            tx_type = ''
            amount = ''
            owner = ''

            for i, text in enumerate(cell_texts):
                # Ticker: 1-5 uppercase letters
                if re.match(r'^[A-Z]{1,5}$', text) and not ticker:
                    ticker = text
                # Date: MM/DD/YYYY pattern
                elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', text) and not tx_date:
                    tx_date = text
                # Type: Purchase/Sale keywords
                elif any(kw in text.lower() for kw in ['purchase', 'sale', 'exchange', 'buy', 'sell']):
                    tx_type = text
                # Amount: dollar sign or range
                elif '$' in text or re.match(r'.*\d+.*-.*\d+', text):
                    amount = text
                # Owner: Self/Spouse/Joint/Child/Dependent
                elif text.lower() in ('self', 'spouse', 'joint', 'child', 'dependent'):
                    owner = text
                # Asset name: longer text without special patterns
                elif len(text) > 5 and not ticker and not asset_name:
                    # Could be asset description
                    # Check if it contains a ticker in parentheses
                    ticker_match = re.search(r'\(([A-Z]{1,5})\)', text)
                    if ticker_match:
                        ticker = ticker_match.group(1)
                        asset_name = text
                    elif not any(c.isdigit() for c in text[:3]):
                        asset_name = text

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
            logger.debug(f"Error parsing transaction row: {e}")

        return None

    def _parse_structured_content(self, soup: BeautifulSoup, filing: Dict) -> List[Dict]:
        """Parse non-table structured content from filing pages."""
        transactions = []

        # Look for sections with transaction data
        sections = soup.find_all(['section', 'div'], class_=re.compile(r'transaction|trade|asset', re.I))

        for section in sections:
            text = section.get_text()
            # Try to extract ticker symbols
            tickers = re.findall(r'\b([A-Z]{1,5})\b', text)
            dates = re.findall(r'(\d{1,2}/\d{1,2}/\d{4})', text)
            amounts = re.findall(r'(\$[\d,]+ - \$[\d,]+|\$[\d,]+)', text)
            types = re.findall(r'(Purchase|Sale|Exchange|Sale \(Full\)|Sale \(Partial\))', text, re.I)

            # Pair them up if we can
            for i, ticker in enumerate(tickers):
                if ticker in ('LLC', 'INC', 'ETF', 'USD', 'USA', 'THE', 'AND', 'FOR'):
                    continue
                tx = {
                    'politician': filing.get('name', 'Unknown'),
                    'ticker': ticker,
                    'asset_description': ticker,
                    'trade_type': types[i] if i < len(types) else 'Unknown',
                    'trade_date': dates[i] if i < len(dates) else filing.get('date', ''),
                    'amount': amounts[i] if i < len(amounts) else '',
                    'owner': 'Self',
                    'disclosure_date': filing.get('date', ''),
                    'chamber': 'Senate',
                    'source_link': filing.get('link', ''),
                }
                transactions.append(tx)

        return transactions

    def scrape(self, days_back: int = 120) -> List[Dict]:
        """
        Full scrape: fetch recent filings, then get transactions from each.
        Returns list of raw transaction dicts.
        """
        all_transactions = []

        filings = self.fetch_recent_filings(days_back)
        logger.info(f"Fetching transactions from {len(filings)} Senate filings...")

        for i, filing in enumerate(filings):
            transactions = self.fetch_filing_transactions(filing)
            all_transactions.extend(transactions)

            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(filings)} filings ({len(all_transactions)} transactions so far)")

            # Rate limiting
            if i < len(filings) - 1:
                time.sleep(0.3)

        logger.info(f"Senate scraper: {len(all_transactions)} total transactions from {len(filings)} filings")
        return all_transactions


class HouseScraper:
    """
    Scrapes the House Clerk Financial Disclosure system.
    URL: https://disclosures-clerk.house.gov/FinancialDisclosure
    """

    BASE_URL = "https://disclosures-clerk.house.gov"
    SEARCH_URL = f"{BASE_URL}/FinancialDisclosure/ViewMemberSearchResult"
    PTR_SEARCH_URL = f"{BASE_URL}/FinancialDisclosure#702SearchResultPTR"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch_recent_ptrs(self, year: Optional[int] = None) -> List[Dict]:
        """
        Fetch recent Periodic Transaction Reports from the House Clerk.
        The House system organizes by filing year.
        """
        filings = []
        if year is None:
            year = datetime.now().year

        try:
            # The House Clerk has a search page for PTRs
            # Try the search endpoint
            search_url = f"{self.BASE_URL}/FinancialDisclosure/ViewMemberSearchResult"
            params = {
                'FilingYear': str(year),
                'State': '',
                'District': '',
                'LastName': '',
                'FilingType': 'P',  # P = Periodic Transaction Report
            }

            logger.info(f"Searching House Clerk for {year} PTR filings...")
            resp = self.session.get(search_url, params=params, timeout=30)
            logger.info(f"House Clerk search response: {resp.status_code}")

            if resp.status_code == 200:
                filings = self._parse_house_results(resp.text)
                logger.info(f"Found {len(filings)} House PTR filings for {year}")

            # Also try the XML/RSS feed if available
            if not filings:
                xml_url = f"{self.BASE_URL}/FinancialDisclosure/ViewMemberSearchResult.xml"
                resp2 = self.session.get(xml_url, params=params, timeout=30)
                if resp2.status_code == 200:
                    filings = self._parse_house_xml(resp2.text)

        except Exception as e:
            logger.error(f"House Clerk search failed: {e}")

        return filings

    def _parse_house_results(self, html: str) -> List[Dict]:
        """Parse House Clerk search results HTML."""
        filings = []
        soup = BeautifulSoup(html, 'html.parser')

        # Find result table
        table = soup.find('table', class_=re.compile(r'result|data|member', re.I))
        if not table:
            table = soup.find('table')

        if not table:
            # Try finding links to PDF filings directly
            links = soup.find_all('a', href=re.compile(r'\.pdf$|ptr|transaction', re.I))
            for link in links:
                text = link.get_text(strip=True)
                href = link.get('href', '')
                if href and text:
                    if not href.startswith('http'):
                        href = f"{self.BASE_URL}{href}"
                    filings.append({
                        'name': text,
                        'link': href,
                        'date': '',
                        'type': 'PTR',
                        'chamber': 'House',
                    })
            return filings

        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 3:
                filing = self._parse_house_row(cells)
                if filing:
                    filings.append(filing)

        return filings

    def _parse_house_row(self, cells: list) -> Optional[Dict]:
        """Parse a House results table row."""
        try:
            name = cells[0].get_text(strip=True)
            # Look for filing link
            link_tag = None
            for cell in cells:
                a = cell.find('a')
                if a and 'href' in a.attrs:
                    link_tag = a
                    break

            link = ''
            if link_tag:
                link = link_tag['href']
                if not link.startswith('http'):
                    link = f"{self.BASE_URL}{link}"

            # District/State
            state = cells[1].get_text(strip=True) if len(cells) > 1 else ''
            date_str = cells[-1].get_text(strip=True) if cells else ''

            # Filing type
            filing_type = ''
            for cell in cells:
                text = cell.get_text(strip=True).lower()
                if 'ptr' in text or 'periodic' in text or 'transaction' in text:
                    filing_type = 'PTR'
                    break

            if name:
                return {
                    'name': name,
                    'link': link,
                    'date': date_str,
                    'state': state,
                    'type': filing_type or 'PTR',
                    'chamber': 'House',
                }
        except Exception:
            pass
        return None

    def _parse_house_xml(self, xml_text: str) -> List[Dict]:
        """Parse House Clerk XML response."""
        filings = []
        soup = BeautifulSoup(xml_text, 'html.parser')

        members = soup.find_all('member') or soup.find_all('filing') or soup.find_all('item')
        for member in members:
            name = member.find('name') or member.find('title')
            link = member.find('link') or member.find('url')
            date = member.find('date') or member.find('filingdate')

            if name:
                filing = {
                    'name': name.get_text(strip=True),
                    'link': link.get_text(strip=True) if link else '',
                    'date': date.get_text(strip=True) if date else '',
                    'type': 'PTR',
                    'chamber': 'House',
                }
                filings.append(filing)

        return filings

    def fetch_filing_transactions(self, filing: Dict) -> List[Dict]:
        """
        Fetch transactions from a House PTR filing.
        House filings are often PDFs, which are harder to parse.
        For HTML filings, we parse the transaction table.
        """
        transactions = []
        link = filing.get('link', '')
        if not link:
            return transactions

        # Skip PDF links (would need pdfplumber/PyPDF2)
        if link.lower().endswith('.pdf'):
            logger.debug(f"Skipping PDF filing: {link}")
            return transactions

        try:
            time.sleep(0.5)
            resp = self.session.get(link, timeout=15)

            if resp.status_code != 200:
                return transactions

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Similar parsing to Senate
            tables = soup.find_all('table')
            for table in tables:
                header_text = table.get_text().lower()[:200]
                if any(kw in header_text for kw in ['ticker', 'transaction', 'asset', 'amount']):
                    rows = table.find_all('tr')[1:]
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 3:
                            tx = self._parse_house_transaction(cells, filing)
                            if tx:
                                transactions.append(tx)

        except Exception as e:
            logger.debug(f"Error fetching House filing: {e}")

        return transactions

    def _parse_house_transaction(self, cells: list, filing: Dict) -> Optional[Dict]:
        """Parse a House transaction row."""
        try:
            cell_texts = [c.get_text(strip=True) for c in cells]

            ticker = ''
            tx_type = ''
            tx_date = ''
            amount = ''
            asset_name = ''

            for text in cell_texts:
                if re.match(r'^[A-Z]{1,5}$', text) and not ticker:
                    ticker = text
                elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', text) and not tx_date:
                    tx_date = text
                elif any(kw in text.lower() for kw in ['purchase', 'sale', 'exchange']):
                    tx_type = text
                elif '$' in text:
                    amount = text
                elif len(text) > 5 and not asset_name:
                    ticker_match = re.search(r'\(([A-Z]{1,5})\)', text)
                    if ticker_match:
                        ticker = ticker_match.group(1)
                    asset_name = text

            if ticker:
                return {
                    'politician': filing.get('name', 'Unknown'),
                    'ticker': ticker,
                    'asset_description': asset_name or ticker,
                    'trade_type': tx_type or 'Unknown',
                    'trade_date': tx_date or filing.get('date', ''),
                    'amount': amount,
                    'owner': 'Self',
                    'disclosure_date': filing.get('date', ''),
                    'chamber': 'House',
                    'state': filing.get('state', ''),
                    'source_link': filing.get('link', ''),
                }
        except Exception:
            pass
        return None

    def scrape(self, days_back: int = 120) -> List[Dict]:
        """Full scrape of House PTR filings."""
        all_transactions = []

        # Search current year and previous year if early in the year
        years = [datetime.now().year]
        if datetime.now().month <= 3:
            years.append(datetime.now().year - 1)

        for year in years:
            filings = self.fetch_recent_ptrs(year)
            logger.info(f"Fetching transactions from {len(filings)} House filings ({year})...")

            for i, filing in enumerate(filings):
                transactions = self.fetch_filing_transactions(filing)
                all_transactions.extend(transactions)

                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{len(filings)} House filings ({len(all_transactions)} transactions)")

                if i < len(filings) - 1:
                    time.sleep(0.3)

        logger.info(f"House scraper: {len(all_transactions)} total transactions")
        return all_transactions


class GovScraper:
    """
    Combined Senate + House government disclosure scraper.
    Manages caching and provides a unified interface.
    """

    def __init__(self):
        self.senate = SenateScraper()
        self.house = HouseScraper()

        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

    def scrape_all(self, days_back: int = 120, use_cache: bool = True) -> List[Dict]:
        """
        Scrape both Senate and House disclosures.
        Uses local file cache (refreshes every 6 hours).
        """
        # Check cache
        if use_cache:
            cached = self._load_cache()
            if cached is not None:
                logger.info(f"Using cached government data: {len(cached)} transactions")
                return cached

        all_trades = []

        # Scrape Senate
        try:
            senate_trades = self.senate.scrape(days_back)
            all_trades.extend(senate_trades)
            logger.info(f"Senate: {len(senate_trades)} transactions")
        except Exception as e:
            logger.error(f"Senate scraper error: {e}")

        # Scrape House
        try:
            house_trades = self.house.scrape(days_back)
            all_trades.extend(house_trades)
            logger.info(f"House: {len(house_trades)} transactions")
        except Exception as e:
            logger.error(f"House scraper error: {e}")

        # Save to cache
        if all_trades:
            self._save_cache(all_trades)

        logger.info(f"Government scraper total: {len(all_trades)} transactions")
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
