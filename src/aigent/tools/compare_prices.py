from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from aigent.schemas import PriceComparisonQuery

_tavily = None

def _get_tavily():
    global _tavily
    if _tavily is None:
        _tavily = TavilySearch(max_results=5, topic="general")
    return _tavily


@tool(args_schema=PriceComparisonQuery)
def compare_prices(product_name: str, max_sources: int = 5) -> str:
    """Compare prices for a product across multiple online retailers."""
    response = _get_tavily().invoke(f"{product_name} price buy")

    if isinstance(response, dict):
        if "error" in response:
            return f"Search error: {response['error']}"
        results = response.get("results", [])
    elif isinstance(response, list):
        results = response
    else:
        return f"Unexpected response format: {type(response)}"

    if not results:
        return "No price information found."

    formatted = []
    for i, r in enumerate(results[:max_sources], 1):
        url = r.get("url", "N/A")
        content = r.get("content", "No description")
        formatted.append(f"{i}. {content}\n   URL: {url}")

    return "\n\n".join(formatted)
