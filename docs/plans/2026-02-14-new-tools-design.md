# New Agent Tools Design

## Overview

Add three new tools to the existing LangGraph agent: price comparison, product reviews, and budget calculator. Expand the Receipt schema to include data from these tools.

## New Tools

### 1. `compare_prices` (`tools/compare_prices.py`)

- **Input:** `PriceComparisonQuery` — `product_name: str`, `max_sources: int = 5`
- **Backend:** Tavily search with query `"{product_name} price buy"`
- **Output:** Formatted list of prices from different retailers with URLs

### 2. `get_reviews` (`tools/get_reviews.py`)

- **Input:** `ReviewQuery` — `product_name: str`, `max_reviews: int = 3`
- **Backend:** Tavily search with query `"{product_name} review rating"`
- **Output:** Formatted review summaries with scores and source URLs

### 3. `calculate_budget` (`tools/calculate_budget.py`)

- **Input:** `BudgetQuery` — `items: list[dict]` (each with `name`, `price`), `tax_rate: float = 0.0`, `budget_limit: float | None = None`
- **Backend:** Pure computation, no API calls
- **Output:** Total, tax, per-item breakdown, within-budget indicator

## Expanded Receipt Schema

```python
class Receipt(BaseModel):
    product_name: str
    price: float
    currency: str = "USD"
    average_rating: float | None = None
    price_range: str | None = None
    recommendation_reason: str | None = None
```

New fields are optional so the agent can return a Receipt even when not all tools were called.

## Agent Changes

- Register all four tools in `create_react_agent(tools=[...])` in `agent.py`
- Export new tools from `tools/__init__.py`
- Agent autonomously decides which tools to call based on user query

## Approach

Three independent tools following the existing `@tool(args_schema=...)` pattern. The ReAct agent orchestrates tool selection — no chaining or meta-tools.
