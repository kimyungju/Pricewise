from unittest.mock import patch, MagicMock
from aigent.tools.get_reviews import get_reviews


def test_get_reviews_returns_formatted_string():
    mock_response = {
        "query": "AirPods Pro review rating",
        "results": [
            {"url": "https://rtings.com/airpods", "content": "AirPods Pro 2 review: 8.4/10 - excellent noise cancellation"},
            {"url": "https://wirecutter.com/airpods", "content": "AirPods Pro 2 - Best wireless earbuds, rated 4.5/5 stars"},
        ],
    }

    mock_instance = MagicMock()
    mock_instance.invoke.return_value = mock_response

    with patch("aigent.tools.get_reviews._get_tavily", return_value=mock_instance):
        result = get_reviews.invoke({"product_name": "AirPods Pro 2", "max_reviews": 2})
        assert isinstance(result, str)
        assert "AirPods" in result or "review" in result.lower()


def test_get_reviews_handles_no_results():
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = {"results": []}

    with patch("aigent.tools.get_reviews._get_tavily", return_value=mock_instance):
        result = get_reviews.invoke({"product_name": "nonexistent product xyz"})
        assert "no reviews" in result.lower()
