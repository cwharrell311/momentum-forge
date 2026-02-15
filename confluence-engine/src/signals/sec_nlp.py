"""
SEC Filing NLP Signal — FinBERT sentiment analysis on 10-K/10-Q filings.

Fetches recent SEC filings from EDGAR, extracts risk factors and MD&A
sections, runs FinBERT sentiment analysis, and produces a signal score.

This is an alternative data signal layer that captures fundamental
sentiment shifts before they show up in price action.

Pipeline:
1. Fetch filing metadata from SEC EDGAR full-text search API
2. Download filing text (10-K annual, 10-Q quarterly)
3. Extract key sections (Risk Factors, MD&A, Business Overview)
4. Run FinBERT sentiment classification on each section
5. Compute aggregate sentiment score + change from prior filing

Requires: transformers, torch (optional — falls back to API-based analysis)

SEC EDGAR API docs: https://efts.sec.gov/LATEST/search-index?q=...
Fair Access: max 10 requests/sec, User-Agent required.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger("forge.sec_nlp")

# SEC EDGAR requires a valid User-Agent with contact info
SEC_USER_AGENT = "MomentumForge/1.0 (research@momentumforge.dev)"
SEC_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
SEC_EDGAR_FILINGS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"

# CIK lookup for common tickers (top holdings)
# Full lookup done via SEC EDGAR company search API
_CIK_CACHE: dict[str, str] = {}


@dataclass
class FilingSentiment:
    """Sentiment analysis result for a single SEC filing."""
    ticker: str
    filing_type: str            # "10-K" or "10-Q"
    filing_date: str
    accession_number: str
    # Section-level sentiment (-1.0 to 1.0)
    risk_factors_sentiment: float
    mda_sentiment: float        # Management Discussion & Analysis
    business_sentiment: float
    # Aggregate
    overall_sentiment: float    # Weighted average across sections
    sentiment_change: float     # Delta from prior filing (0 if first)
    # Metadata
    word_count: int
    negative_pct: float         # % of sentences classified negative
    positive_pct: float


@dataclass
class SECSignal:
    """Trading signal derived from SEC filing sentiment."""
    ticker: str
    score: float                # -1.0 (very negative) to 1.0 (very positive)
    confidence: float           # 0-1
    filing_date: str
    filing_type: str
    reason: str


class FinBERTAnalyzer:
    """
    FinBERT-based sentiment analysis for financial text.

    Uses the ProsusAI/finbert model from HuggingFace. Falls back to
    keyword-based analysis if transformers/torch not available.
    """

    def __init__(self):
        self._pipeline = None
        self._available = None

    def _ensure_loaded(self):
        """Lazy-load the FinBERT model on first use."""
        if self._available is not None:
            return self._available

        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                truncation=True,
                max_length=512,
            )
            self._available = True
            log.info("FinBERT model loaded successfully")
        except Exception as e:
            log.warning("FinBERT not available (%s), using keyword fallback", e)
            self._available = False

        return self._available

    def analyze_text(self, text: str) -> dict[str, float]:
        """
        Analyze financial text sentiment.

        Returns:
            {"positive": 0.x, "negative": 0.x, "neutral": 0.x, "score": -1 to 1}
        """
        if not text or len(text.strip()) < 20:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "score": 0.0}

        if self._ensure_loaded() and self._pipeline is not None:
            return self._analyze_finbert(text)
        else:
            return self._analyze_keywords(text)

    def _analyze_finbert(self, text: str) -> dict[str, float]:
        """Run FinBERT on text, chunking if necessary."""
        # Split into sentences for better granularity
        sentences = _split_sentences(text)
        if not sentences:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "score": 0.0}

        # Process in batches (FinBERT has 512 token limit)
        batch_size = 32
        pos_total = neg_total = neu_total = 0.0
        count = 0

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            try:
                results = self._pipeline(batch)
                for r in results:
                    label = r["label"].lower()
                    score = r["score"]
                    if label == "positive":
                        pos_total += score
                    elif label == "negative":
                        neg_total += score
                    else:
                        neu_total += score
                    count += 1
            except Exception as e:
                log.debug("FinBERT batch failed: %s", e)
                continue

        if count == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "score": 0.0}

        pos = pos_total / count
        neg = neg_total / count
        neu = neu_total / count
        # Score: -1 (all negative) to +1 (all positive)
        score = pos - neg

        return {"positive": pos, "negative": neg, "neutral": neu, "score": score}

    def _analyze_keywords(self, text: str) -> dict[str, float]:
        """
        Keyword-based sentiment fallback when FinBERT is unavailable.

        Uses Loughran-McDonald financial sentiment dictionary (simplified).
        """
        text_lower = text.lower()
        words = text_lower.split()
        total = len(words)
        if total == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "score": 0.0}

        # Loughran-McDonald negative words (top 50 most discriminating)
        negative_words = {
            "loss", "losses", "decline", "declined", "adverse", "adversely",
            "risk", "risks", "impairment", "impaired", "litigation",
            "default", "defaults", "failure", "failed", "violation",
            "terminate", "terminated", "downgrade", "downgraded",
            "restructuring", "bankruptcy", "insolvent", "insolvency",
            "deficit", "deficiency", "unfavorable", "unfavourable",
            "deteriorate", "deteriorated", "deterioration",
            "uncertain", "uncertainty", "threat", "threats",
            "penalty", "penalties", "suspend", "suspended",
            "weakness", "weaknesses", "unable", "inability",
            "exposure", "volatility", "downturn", "recession",
            "write-off", "writedown", "impair", "curtail",
        }

        positive_words = {
            "growth", "increase", "increased", "improvement", "improved",
            "profit", "profitable", "gain", "gains", "revenue",
            "strong", "strength", "favorable", "favourable",
            "opportunity", "opportunities", "innovation",
            "exceed", "exceeded", "exceeding", "outperform",
            "expansion", "expanded", "enhance", "enhanced",
            "success", "successful", "achievement", "achieved",
            "momentum", "accelerate", "accelerated", "robust",
            "efficient", "efficiency", "dividend", "upgrade",
            "optimistic", "confident", "confidence", "milestone",
        }

        neg_count = sum(1 for w in words if w in negative_words)
        pos_count = sum(1 for w in words if w in positive_words)

        neg_pct = neg_count / total
        pos_pct = pos_count / total
        neu_pct = 1.0 - neg_pct - pos_pct

        score = pos_pct - neg_pct
        # Normalize to -1..1 range (typical filing has ~2-5% sentiment words)
        score = np.clip(score * 20.0, -1.0, 1.0)

        return {
            "positive": pos_pct,
            "negative": neg_pct,
            "neutral": max(0, neu_pct),
            "score": float(score),
        }


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering very short ones."""
    # Simple sentence splitter
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


class SECEdgarClient:
    """
    SEC EDGAR API client for fetching filing metadata and text.

    Respects SEC fair access: max 10 req/sec, User-Agent required.
    """

    def __init__(self, user_agent: str = SEC_USER_AGENT):
        self.user_agent = user_agent
        self._last_request = 0.0

    def _throttle(self):
        """Ensure max 10 requests/second to SEC EDGAR."""
        elapsed = time.time() - self._last_request
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self._last_request = time.time()

    def get_cik(self, ticker: str) -> str | None:
        """Look up CIK number for a ticker symbol."""
        if ticker in _CIK_CACHE:
            return _CIK_CACHE[ticker]

        import httpx
        self._throttle()

        try:
            resp = httpx.get(
                "https://www.sec.gov/cgi-bin/browse-edgar",
                params={
                    "action": "getcompany",
                    "company": ticker,
                    "CIK": ticker,
                    "type": "10-K",
                    "dateb": "",
                    "owner": "include",
                    "count": "1",
                    "search_text": "",
                    "action": "getcompany",
                    "output": "atom",
                },
                headers={"User-Agent": self.user_agent},
                timeout=10,
            )
            # Parse CIK from atom feed
            match = re.search(r'CIK=(\d+)', resp.text)
            if match:
                cik = match.group(1)
                _CIK_CACHE[ticker] = cik
                return cik
        except Exception as e:
            log.debug("CIK lookup failed for %s: %s", ticker, e)

        return None

    def get_recent_filings(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Fetch recent SEC filings for a ticker.

        Returns list of filing metadata dicts with keys:
            accession_number, filing_date, filing_type, primary_doc_url
        """
        if filing_types is None:
            filing_types = ["10-K", "10-Q"]

        import httpx
        self._throttle()

        try:
            # Use EDGAR full-text search API
            resp = httpx.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={
                    "q": f'"{ticker}"',
                    "dateRange": "custom",
                    "startdt": "2023-01-01",
                    "enddt": "2026-12-31",
                    "forms": ",".join(filing_types),
                },
                headers={"User-Agent": self.user_agent},
                timeout=15,
            )

            if resp.status_code != 200:
                log.debug("EDGAR search returned %d for %s", resp.status_code, ticker)
                return []

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            filings = []
            for hit in hits[:limit]:
                source = hit.get("_source", {})
                filings.append({
                    "accession_number": source.get("file_num", ""),
                    "filing_date": source.get("file_date", ""),
                    "filing_type": source.get("form_type", ""),
                    "entity_name": source.get("entity_name", ""),
                    "file_url": source.get("file_url", ""),
                })

            return filings

        except Exception as e:
            log.debug("EDGAR filing fetch failed for %s: %s", ticker, e)
            return []

    def fetch_filing_text(self, file_url: str, max_chars: int = 100_000) -> str:
        """Fetch the raw text of a filing, truncated to max_chars."""
        import httpx
        self._throttle()

        try:
            resp = httpx.get(
                file_url,
                headers={"User-Agent": self.user_agent},
                timeout=30,
            )
            if resp.status_code == 200:
                # Strip HTML tags for cleaner text
                text = re.sub(r'<[^>]+>', ' ', resp.text)
                text = re.sub(r'\s+', ' ', text)
                return text[:max_chars]
        except Exception as e:
            log.debug("Filing text fetch failed: %s", e)

        return ""


def extract_sections(filing_text: str) -> dict[str, str]:
    """
    Extract key sections from a 10-K/10-Q filing.

    Looks for:
    - Item 1A: Risk Factors
    - Item 7: Management's Discussion and Analysis (MD&A)
    - Item 1: Business
    """
    sections = {
        "risk_factors": "",
        "mda": "",
        "business": "",
    }

    text = filing_text

    # Risk Factors (Item 1A)
    risk_match = re.search(
        r'Item\s+1A[.\s]*Risk\s+Factors(.*?)(?=Item\s+1B|Item\s+2)',
        text, re.IGNORECASE | re.DOTALL,
    )
    if risk_match:
        sections["risk_factors"] = risk_match.group(1)[:30000]

    # MD&A (Item 7)
    mda_match = re.search(
        r"Item\s+7[.\s]*Management.s\s+Discussion(.*?)(?=Item\s+7A|Item\s+8)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if mda_match:
        sections["mda"] = mda_match.group(1)[:30000]

    # Business (Item 1)
    biz_match = re.search(
        r'Item\s+1[.\s]*Business(.*?)(?=Item\s+1A|Item\s+2)',
        text, re.IGNORECASE | re.DOTALL,
    )
    if biz_match:
        sections["business"] = biz_match.group(1)[:20000]

    return sections


def analyze_filing(
    ticker: str,
    filing_text: str,
    filing_type: str,
    filing_date: str,
    accession_number: str,
    analyzer: FinBERTAnalyzer | None = None,
    prior_sentiment: float | None = None,
) -> FilingSentiment:
    """
    Run sentiment analysis on a single SEC filing.

    Args:
        ticker: Stock ticker.
        filing_text: Raw filing text.
        filing_type: "10-K" or "10-Q".
        filing_date: Filing date string.
        accession_number: SEC accession number.
        analyzer: FinBERT analyzer instance (reuse for efficiency).
        prior_sentiment: Previous filing's overall sentiment for delta calc.

    Returns:
        FilingSentiment with section-level and aggregate scores.
    """
    if analyzer is None:
        analyzer = FinBERTAnalyzer()

    sections = extract_sections(filing_text)

    # Analyze each section
    risk_result = analyzer.analyze_text(sections["risk_factors"])
    mda_result = analyzer.analyze_text(sections["mda"])
    biz_result = analyzer.analyze_text(sections["business"])

    # Weighted aggregate: MD&A matters most, risk factors second
    weights = {"risk_factors": 0.30, "mda": 0.50, "business": 0.20}
    overall = (
        risk_result["score"] * weights["risk_factors"]
        + mda_result["score"] * weights["mda"]
        + biz_result["score"] * weights["business"]
    )

    # Sentiment change from prior filing
    change = 0.0
    if prior_sentiment is not None:
        change = overall - prior_sentiment

    word_count = len(filing_text.split())

    return FilingSentiment(
        ticker=ticker,
        filing_type=filing_type,
        filing_date=filing_date,
        accession_number=accession_number,
        risk_factors_sentiment=risk_result["score"],
        mda_sentiment=mda_result["score"],
        business_sentiment=biz_result["score"],
        overall_sentiment=overall,
        sentiment_change=change,
        word_count=word_count,
        negative_pct=risk_result["negative"],
        positive_pct=mda_result["positive"],
    )


def get_sec_signal(
    ticker: str,
    edgar: SECEdgarClient | None = None,
    analyzer: FinBERTAnalyzer | None = None,
) -> SECSignal | None:
    """
    Get the current SEC filing sentiment signal for a ticker.

    Fetches the most recent 10-K/10-Q, analyzes sentiment, and returns
    a trading signal based on sentiment level and change.

    Args:
        ticker: Stock symbol.
        edgar: EDGAR client instance.
        analyzer: FinBERT analyzer instance.

    Returns:
        SECSignal or None if no filings available.
    """
    if edgar is None:
        edgar = SECEdgarClient()
    if analyzer is None:
        analyzer = FinBERTAnalyzer()

    filings = edgar.get_recent_filings(ticker, limit=2)
    if not filings:
        return None

    latest = filings[0]
    file_url = latest.get("file_url", "")
    if not file_url:
        return None

    # Analyze latest filing
    text = edgar.fetch_filing_text(file_url)
    if not text:
        return None

    # Analyze prior filing for delta
    prior_sentiment = None
    if len(filings) > 1:
        prior_url = filings[1].get("file_url", "")
        if prior_url:
            prior_text = edgar.fetch_filing_text(prior_url)
            if prior_text:
                prior = analyze_filing(
                    ticker, prior_text, filings[1]["filing_type"],
                    filings[1]["filing_date"], filings[1].get("accession_number", ""),
                    analyzer,
                )
                prior_sentiment = prior.overall_sentiment

    result = analyze_filing(
        ticker, text, latest["filing_type"],
        latest["filing_date"], latest.get("accession_number", ""),
        analyzer, prior_sentiment,
    )

    # Convert to trading signal
    # Score combines level + change (change is more actionable)
    score = result.overall_sentiment * 0.4 + result.sentiment_change * 0.6
    score = float(np.clip(score, -1.0, 1.0))

    # Confidence based on word count (more text = more reliable)
    confidence = min(1.0, result.word_count / 10000)

    reason_parts = [f"SEC {result.filing_type} ({result.filing_date})"]
    reason_parts.append(f"sentiment={result.overall_sentiment:+.2f}")
    if result.sentiment_change != 0:
        reason_parts.append(f"delta={result.sentiment_change:+.2f}")

    return SECSignal(
        ticker=ticker,
        score=score,
        confidence=confidence,
        filing_date=result.filing_date,
        filing_type=result.filing_type,
        reason=", ".join(reason_parts),
    )
