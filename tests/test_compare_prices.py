from unittest.mock import patch, MagicMock
from aigent.tools.compare_prices import compare_prices


def test_compare_prices_returns_formatted_string():
    mock_response = {
        "query": "Sony WH-1000XM5 price buy",
        "results": [
            {"url": "https://amazon.com/sony", "content": "Sony WH-1000XM5 - $298 at Amazon"},
            {"url": "https://bestbuy.com/sony", "content": "Sony WH-1000XM5 - $329 at Best Buy"},
        ],
    }

    mock_instance = MagicMock()
    mock_instance.invoke.return_value = mock_response

    with patch("aigent.tools.compare_prices._get_tavily", return_value=mock_instance):
        result = compare_prices.invoke({"product_name": "Sony WH-1000XM5", "max_sources": 2})
        assert isinstance(result, str)
        assert "Amazon" in result or "Sony" in result


def test_compare_prices_handles_error():
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = {"error": "API error"}

    with patch("aigent.tools.compare_prices._get_tavily", return_value=mock_instance):
        result = compare_prices.invoke({"product_name": "test"})
        assert "error" in result.lower()
