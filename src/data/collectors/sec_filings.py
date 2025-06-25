"""
SEC Filing Collector

Collects and processes SEC filings (10-Q, 10-K) for G-SIB banks.
Uses the SEC EDGAR database API to fetch filings and parse XML content.
"""

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import requests
import time
from loguru import logger

from ..schemas.data_models import SECFiling, DocumentType
from ...config.settings import settings


class SECFilingCollector:
    """Collector for SEC filings from EDGAR database."""
    
    # G-SIB bank CIKs (Central Index Keys)
    GSIB_BANKS = {
        "JPM": {"cik": "0000019617", "name": "JPMorgan Chase & Co."},
        "BAC": {"cik": "0000070858", "name": "Bank of America Corporation"},
        "WFC": {"cik": "0000072971", "name": "Wells Fargo & Company"},
        "C": {"cik": "0000831001", "name": "Citigroup Inc."},
        "GS": {"cik": "0000886982", "name": "The Goldman Sachs Group, Inc."},
        "MS": {"cik": "0000895421", "name": "Morgan Stanley"},
        "USB": {"cik": "0000036104", "name": "U.S. Bancorp"},
        "PNC": {"cik": "0000713676", "name": "The PNC Financial Services Group, Inc."},
        "TD": {"cik": "0000947263", "name": "The Toronto-Dominion Bank"},
        "RY": {"cik": "0000000000", "name": "Royal Bank of Canada"},  # Placeholder
    }
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize SEC filing collector."""
        self.output_dir = Path(output_dir or settings.sec_filings_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # SEC API configuration
        self.base_url = "https://data.sec.gov"
        self.headers = {
            "User-Agent": "LLM Finance Leaderboard research@yourorg.com",
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }
        
        # Rate limiting (SEC allows 10 requests per second)
        self.request_delay = 0.1
        self.last_request_time = 0
    
    def _rate_limit(self) -> None:
        """Implement rate limiting for SEC API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str) -> requests.Response:
        """Make rate-limited request to SEC API."""
        self._rate_limit()
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response
    
    def get_company_filings(
        self, 
        cik: str, 
        filing_type: str = "10-Q",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get filings for a specific company.
        
        Args:
            cik: Central Index Key
            filing_type: Type of filing (10-Q, 10-K, etc.)
            start_date: Start date for filing search
            end_date: End date for filing search
            limit: Maximum number of filings to return
            
        Returns:
            List of filing metadata dictionaries
        """
        # Format CIK (pad with zeros to 10 digits)
        formatted_cik = cik.zfill(10)
        
        # Build API URL
        url = f"{self.base_url}/submissions/CIK{formatted_cik}.json"
        
        try:
            response = self._make_request(url)
            data = response.json()
            
            # Extract filings
            filings = data.get("filings", {}).get("recent", {})
            
            # Filter by filing type and date range
            filtered_filings = []
            for i, form in enumerate(filings.get("form", [])):
                if form != filing_type:
                    continue
                
                filing_date_str = filings.get("filingDate", [])[i]
                filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
                
                # Apply date filters
                if start_date and filing_date < start_date:
                    continue
                if end_date and filing_date > end_date:
                    continue
                
                filing_info = {
                    "accessionNumber": filings.get("accessionNumber", [])[i],
                    "filingDate": filing_date_str,
                    "reportDate": filings.get("reportDate", [])[i],
                    "acceptanceDateTime": filings.get("acceptanceDateTime", [])[i],
                    "form": form,
                    "fileNumber": filings.get("fileNumber", [])[i],
                    "filmNumber": filings.get("filmNumber", [])[i],
                    "items": filings.get("items", [])[i] if i < len(filings.get("items", [])) else "",
                    "size": filings.get("size", [])[i],
                    "isXBRL": filings.get("isXBRL", [])[i],
                    "isInlineXBRL": filings.get("isInlineXBRL", [])[i],
                    "primaryDocument": filings.get("primaryDocument", [])[i],
                    "primaryDocDescription": filings.get("primaryDocDescription", [])[i],
                }
                
                filtered_filings.append(filing_info)
                
                if len(filtered_filings) >= limit:
                    break
            
            logger.info(f"Found {len(filtered_filings)} {filing_type} filings for CIK {cik}")
            return filtered_filings
            
        except Exception as e:
            logger.error(f"Error fetching filings for CIK {cik}: {e}")
            return []
    
    def download_filing_content(self, cik: str, accession_number: str, primary_document: str) -> Optional[str]:
        """
        Download the content of a specific filing.
        
        Args:
            cik: Central Index Key
            accession_number: SEC accession number
            primary_document: Primary document filename
            
        Returns:
            Filing content as string, or None if failed
        """
        # Format CIK and accession number
        formatted_cik = cik.zfill(10)
        formatted_accession = accession_number.replace("-", "")
        
        # Build document URL
        url = f"{self.base_url}/Archives/edgar/data/{int(cik)}/{formatted_accession}/{primary_document}"
        
        try:
            response = self._make_request(url)
            return response.text
        except Exception as e:
            logger.error(f"Error downloading filing content: {e}")
            return None
    
    def parse_filing_content(self, content: str, filing_type: str) -> Dict[str, str]:
        """
        Parse filing content to extract key sections.
        
        Args:
            content: Raw filing content
            filing_type: Type of filing (10-Q, 10-K)
            
        Returns:
            Dictionary of parsed sections
        """
        sections = {}
        
        try:
            # Try to parse as XML first
            if content.strip().startswith('<?xml'):
                sections.update(self._parse_xml_filing(content))
            else:
                # Parse as HTML/text
                sections.update(self._parse_text_filing(content, filing_type))
                
        except Exception as e:
            logger.warning(f"Error parsing filing content: {e}")
            # Fallback to basic text extraction
            sections["full_text"] = content
        
        return sections
    
    def _parse_xml_filing(self, content: str) -> Dict[str, str]:
        """Parse XML-based filing content."""
        sections = {}
        
        try:
            root = ET.fromstring(content)
            
            # Extract common XBRL elements
            namespaces = {
                'xbrli': 'http://www.xbrl.org/2003/instance',
                'us-gaap': 'http://fasb.org/us-gaap/2021-01-31',
                'dei': 'http://xbrl.sec.gov/dei/2021-01-31'
            }
            
            # Extract key financial metrics
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                    sections[tag_name] = elem.text.strip()
            
        except ET.ParseError as e:
            logger.warning(f"XML parsing error: {e}")
            sections["raw_content"] = content
        
        return sections
    
    def _parse_text_filing(self, content: str, filing_type: str) -> Dict[str, str]:
        """Parse text-based filing content."""
        sections = {}
        
        # Common section patterns for 10-Q and 10-K filings
        if filing_type == "10-Q":
            patterns = {
                "financial_statements": r"CONDENSED CONSOLIDATED STATEMENTS.*?(?=ITEM|$)",
                "management_discussion": r"MANAGEMENT'S DISCUSSION AND ANALYSIS.*?(?=ITEM|$)",
                "controls_procedures": r"CONTROLS AND PROCEDURES.*?(?=ITEM|$)",
            }
        else:  # 10-K
            patterns = {
                "business": r"ITEM 1\..*?BUSINESS.*?(?=ITEM|$)",
                "risk_factors": r"ITEM 1A\..*?RISK FACTORS.*?(?=ITEM|$)",
                "financial_statements": r"CONSOLIDATED STATEMENTS.*?(?=ITEM|$)",
                "management_discussion": r"MANAGEMENT'S DISCUSSION AND ANALYSIS.*?(?=ITEM|$)",
            }
        
        # Extract sections using regex
        for section_name, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = match.group(0).strip()
        
        # If no sections found, store full content
        if not sections:
            sections["full_text"] = content
        
        return sections
    
    def collect_filings(
        self,
        tickers: Optional[List[str]] = None,
        filing_types: List[str] = ["10-Q", "10-K"],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit_per_company: int = 20
    ) -> List[SECFiling]:
        """
        Collect SEC filings for specified companies.
        
        Args:
            tickers: List of ticker symbols (if None, uses all G-SIB banks)
            filing_types: Types of filings to collect
            start_date: Start date for collection
            end_date: End date for collection
            limit_per_company: Maximum filings per company
            
        Returns:
            List of SECFiling objects
        """
        if tickers is None:
            tickers = list(self.GSIB_BANKS.keys())
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 2)  # Default: 2 years
        
        if end_date is None:
            end_date = datetime.now()
        
        collected_filings = []
        
        for ticker in tickers:
            if ticker not in self.GSIB_BANKS:
                logger.warning(f"Unknown ticker: {ticker}")
                continue
            
            bank_info = self.GSIB_BANKS[ticker]
            cik = bank_info["cik"]
            company_name = bank_info["name"]
            
            logger.info(f"Collecting filings for {ticker} ({company_name})")
            
            for filing_type in filing_types:
                # Get filing metadata
                filings_metadata = self.get_company_filings(
                    cik=cik,
                    filing_type=filing_type,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit_per_company
                )
                
                for filing_meta in filings_metadata:
                    try:
                        # Download filing content
                        content = self.download_filing_content(
                            cik=cik,
                            accession_number=filing_meta["accessionNumber"],
                            primary_document=filing_meta["primaryDocument"]
                        )
                        
                        if not content:
                            continue
                        
                        # Parse content
                        parsed_sections = self.parse_filing_content(content, filing_type)
                        
                        # Create SECFiling object
                        filing = SECFiling(
                            title=f"{company_name} {filing_type} - {filing_meta['filingDate']}",
                            content=content,
                            document_type=DocumentType.SEC_10Q if filing_type == "10-Q" else DocumentType.SEC_10K,
                            cik=cik,
                            ticker=ticker,
                            company_name=company_name,
                            filing_type=filing_type,
                            filing_date=datetime.strptime(filing_meta["filingDate"], "%Y-%m-%d"),
                            period_end_date=datetime.strptime(filing_meta["reportDate"], "%Y-%m-%d"),
                            fiscal_year=datetime.strptime(filing_meta["reportDate"], "%Y-%m-%d").year,
                            fiscal_quarter=((datetime.strptime(filing_meta["reportDate"], "%Y-%m-%d").month - 1) // 3) + 1 if filing_type == "10-Q" else None,
                            accession_number=filing_meta["accessionNumber"],
                            metadata={
                                "parsed_sections": parsed_sections,
                                "file_size": filing_meta["size"],
                                "is_xbrl": filing_meta["isXBRL"],
                                "primary_document": filing_meta["primaryDocument"],
                            }
                        )
                        
                        collected_filings.append(filing)
                        
                        # Save to file
                        self._save_filing(filing)
                        
                        logger.info(f"Collected {filing_type} filing for {ticker}: {filing_meta['filingDate']}")
                        
                    except Exception as e:
                        logger.error(f"Error processing filing {filing_meta['accessionNumber']}: {e}")
                        continue
        
        logger.info(f"Total filings collected: {len(collected_filings)}")
        return collected_filings
    
    def _save_filing(self, filing: SECFiling) -> None:
        """Save filing to local storage."""
        filename = f"{filing.ticker}_{filing.filing_type}_{filing.filing_date.strftime('%Y%m%d')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(filing.json(indent=2))
        
        logger.debug(f"Saved filing to {filepath}")