from pydantic import BaseModel, Field


class ProductQuery(BaseModel):
    """Input schema for the SearchProduct tool."""
    query: str = Field(description="The product search term")
    max_results: int = Field(default=3, description="Maximum number of results to return")


class PriceComparisonQuery(BaseModel):
    """Input schema for the compare_prices tool."""
    product_name: str = Field(description="Name of the product to compare prices for")
    max_sources: int = Field(default=5, description="Maximum number of price sources to return")


class ReviewQuery(BaseModel):
    """Input schema for the get_reviews tool."""
    product_name: str = Field(description="Name of the product to get reviews for")
    max_reviews: int = Field(default=3, description="Maximum number of review sources to return")


class Receipt(BaseModel):
    """Structured output schema for the agent's final response."""
    product_name: str = Field(description="Name of the recommended product")
    price: float = Field(description="Price of the product")
    currency: str = Field(default="USD", description="Currency code")
    average_rating: float | None = Field(default=None, description="Average rating from reviews")
    price_range: str | None = Field(default=None, description="Price range across retailers, e.g. '$49 - $79'")
    recommendation_reason: str | None = Field(default=None, description="Why this product is recommended")


class BudgetItem(BaseModel):
    """A single item in a budget calculation."""
    name: str = Field(description="Product name")
    price: float = Field(description="Price of the product")


class BudgetQuery(BaseModel):
    """Input schema for the calculate_budget tool."""
    items: list[BudgetItem] = Field(description="List of items with name and price")
    tax_rate: float = Field(default=0.0, description="Tax rate as a decimal (e.g., 0.08 for 8%)")
    budget_limit: float | None = Field(default=None, description="Optional budget limit to check against")
