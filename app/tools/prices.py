"""Price tool for querying market data."""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class PriceTool:
    """Tool for querying price data from prices.json."""
    
    def __init__(self, prices_path: str = "prices_stub/prices.json"):
        """Initialize the price tool."""
        self.prices_path = Path(prices_path)
        self.prices_data: Dict[str, List[Dict]] = {}
        self._load_prices()
    
    def _load_prices(self):
        """Load prices from JSON file."""
        if self.prices_path.exists():
            with open(self.prices_path, 'r') as f:
                self.prices_data = json.load(f)
        else:
            raise FileNotFoundError(f"Prices file not found at {self.prices_path}")
    
    def get_latest_price(self, ticker: str) -> Optional[Dict[str, any]]:
        """Get the most recent price for a ticker.
        
        Args:
            ticker: Ticker symbol (e.g., 'AAPL', 'MSFT')
            
        Returns:
            Dict with date and close price, or None if not found
        """
        ticker = ticker.upper()
        if ticker not in self.prices_data:
            return None
        
        # Get the most recent price (last in the list)
        prices = self.prices_data[ticker]
        if not prices:
            return None
        
        latest = max(prices, key=lambda x: x['date'])
        return {
            "ticker": ticker,
            "date": latest['date'],
            "close": latest['close']
        }
    
    def get_price_at_date(self, ticker: str, date: str) -> Optional[Dict[str, any]]:
        """Get price for a specific date.
        
        Args:
            ticker: Ticker symbol
            date: Date string in YYYY-MM-DD format
            
        Returns:
            Dict with date and close price, or None if not found
        """
        ticker = ticker.upper()
        if ticker not in self.prices_data:
            return None
        
        for price_entry in self.prices_data[ticker]:
            if price_entry['date'] == date:
                return {
                    "ticker": ticker,
                    "date": price_entry['date'],
                    "close": price_entry['close']
                }
        return None
    
    def compare_tickers(self, ticker1: str, ticker2: str, days: Optional[int] = None) -> Optional[Dict[str, any]]:
        """Compare performance of two tickers.
        
        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            days: Number of days to look back (uses all available if None)
            
        Returns:
            Dict with comparison data or None if tickers not found
        """
        ticker1 = ticker1.upper()
        ticker2 = ticker2.upper()
        
        if ticker1 not in self.prices_data or ticker2 not in self.prices_data:
            return None
        
        prices1 = sorted(self.prices_data[ticker1], key=lambda x: x['date'])
        prices2 = sorted(self.prices_data[ticker2], key=lambda x: x['date'])
        
        if not prices1 or not prices2:
            return None
        
        # Limit to specified days if provided
        if days:
            prices1 = prices1[-days:]
            prices2 = prices2[-days:]
        
        # Calculate returns
        start_price1 = prices1[0]['close']
        end_price1 = prices1[-1]['close']
        return1 = ((end_price1 - start_price1) / start_price1) * 100
        
        start_price2 = prices2[0]['close']
        end_price2 = prices2[-1]['close']
        return2 = ((end_price2 - start_price2) / start_price2) * 100
        
        return {
            ticker1: {
                "start_date": prices1[0]['date'],
                "end_date": prices1[-1]['date'],
                "start_price": start_price1,
                "end_price": end_price1,
                "return_pct": round(return1, 2),
                "prices": prices1
            },
            ticker2: {
                "start_date": prices2[0]['date'],
                "end_date": prices2[-1]['date'],
                "start_price": start_price2,
                "end_price": end_price2,
                "return_pct": round(return2, 2),
                "prices": prices2
            },
            "relative_performance": round(return1 - return2, 2)
        }
    
    def get_all_tickers(self) -> List[str]:
        """Get list of all available tickers."""
        return list(self.prices_data.keys())
    
    def query_prices(self, query: str) -> str:
        """Natural language query interface for LangChain tool integration.

        Args:
            query: Natural language query about prices
            
        Returns:
            String response with price information
        """
        query_lower = query.lower()

        # Extract tickers (common patterns)
        tickers = []
        for ticker in self.get_all_tickers():
            if ticker.lower() in query_lower:
                tickers.append(ticker)

        if not tickers:
            return "No recognized tickers found in query."

        # Look for explicit date references before falling back to keyword heuristics
        date_str = self._extract_date(query)
        if date_str:
            results = []
            for ticker in tickers:
                price_data = self.get_price_at_date(ticker, date_str)
                if price_data:
                    results.append(f"{ticker}: ${price_data['close']} on {price_data['date']}")
            if results:
                return ", ".join(results)
            return f"No price data found for {', '.join(tickers)} on {date_str}."

        # Check for comparison queries
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'performance']):
            if len(tickers) >= 2:
                # Look for day references
                days = None
                if '10 day' in query_lower or 'last 10' in query_lower:
                    days = 10
                
                result = self.compare_tickers(tickers[0], tickers[1], days)
                if result:
                    return self._format_comparison_result(result, tickers[0], tickers[1])
        
        # Check for latest/recent/close queries
        if any(word in query_lower for word in ['latest', 'recent', 'last', 'close', 'current']):
            results = []
            for ticker in tickers:
                price_data = self.get_latest_price(ticker)
                if price_data:
                    results.append(f"{ticker}: ${price_data['close']} on {price_data['date']}")
            return "Latest prices: " + ", ".join(results) if results else "No price data found."
        
        # Default: return latest for all found tickers
        results = []
        for ticker in tickers:
            price_data = self.get_latest_price(ticker)
            if price_data:
                results.append(f"{ticker}: ${price_data['close']} on {price_data['date']}")
        return ", ".join(results) if results else "No price data found."

    def _format_comparison_result(self, result: Dict, ticker1: str, ticker2: str) -> str:
        """Format comparison result as a string."""
        t1_data = result[ticker1]
        t2_data = result[ticker2]
        
        output = f"Performance comparison:\n"
        output += f"{ticker1}: {t1_data['start_price']:.2f} ({t1_data['start_date']}) → {t1_data['end_price']:.2f} ({t1_data['end_date']}), "
        output += f"Return: {t1_data['return_pct']:.2f}%\n"
        output += f"{ticker2}: {t2_data['start_price']:.2f} ({t2_data['start_date']}) → {t2_data['end_price']:.2f} ({t2_data['end_date']}), "
        output += f"Return: {t2_data['return_pct']:.2f}%\n"
        output += f"Relative performance ({ticker1} vs {ticker2}): {result['relative_performance']:.2f}%"

        return output

    def _extract_date(self, query: str) -> Optional[str]:
        """Extract an ISO-format date (YYYY-MM-DD) from the query, if present."""
        iso_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
        if iso_match:
            date_str = iso_match.group(1)
            try:
                # Validate the date
                datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                return None
        return None
