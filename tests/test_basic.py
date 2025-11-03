"""Basic tests for RAG agent components."""
import pytest
from app.tools.prices import PriceTool


def test_price_tool_latest():
    """Test price tool latest price retrieval."""
    price_tool = PriceTool()
    
    # Test getting latest price for MSFT
    result = price_tool.get_latest_price("MSFT")
    assert result is not None
    assert result["ticker"] == "MSFT"
    assert "close" in result
    assert "date" in result


def test_price_tool_comparison():
    """Test price tool comparison."""
    price_tool = PriceTool()
    
    # Test comparison
    result = price_tool.compare_tickers("SPY", "QQQ")
    assert result is not None
    assert "SPY" in result
    assert "QQQ" in result
    assert "relative_performance" in result


def test_price_tool_query():
    """Test price tool natural language query."""
    price_tool = PriceTool()
    
    # Test query
    result = price_tool.query_prices("What is the latest price for MSFT?")
    assert "MSFT" in result
    assert "$" in result


def test_price_tool_date_query():
    """Test price tool query for a specific date."""
    price_tool = PriceTool()

    result = price_tool.query_prices("What was the SPY close on 2025-06-01?")
    assert "2025-06-01" in result
    assert "$525.2" in result or "$525.20" in result


def test_price_tool_invalid_ticker():
    """Test price tool with invalid ticker."""
    price_tool = PriceTool()
    
    result = price_tool.get_latest_price("INVALID")
    assert result is None

