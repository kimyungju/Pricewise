from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from aigent.schemas import ReviewQuery

_tavily = None

def _get_tavily():
    global _tavily
    if _tavily is None:
        _tavily = TavilySearch(max_results=5, topic="general")
    return _tavily


@tool(args_schema=ReviewQuery)
def get_reviews(product_name: str, max_reviews: int = 3) -> str:
    """Fetch product reviews and ratings from the web."""
    response = _get_tavily().invoke(f"{product_name} review rating")

    if isinstance(response, dict):
        if "error" in response:
            return f"Search error: {response['error']}"
        results = response.get("results", [])
    elif isinstance(response, list):
        results = response
    else:
        return f"Unexpected response format: {type(response)}"

    if not results:
        return "No reviews found for this product."

    formatted = []
    for i, r in enumerate(results[:max_reviews], 1):
        url = r.get("url", "N/A")
        content = r.get("content", "No description")
        formatted.append(f"{i}. {content}\n   Source: {url}")

    return "\n\n".join(formatted)
