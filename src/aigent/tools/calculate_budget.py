from langchain_core.tools import tool
from aigent.schemas import BudgetQuery


@tool(args_schema=BudgetQuery)
def calculate_budget(items: list, tax_rate: float = 0.0, budget_limit: float | None = None) -> str:
    """Calculate total cost with tax and check against an optional budget limit."""
    if not items:
        return "No items provided."

    lines = []
    subtotal = 0.0

    for item in items:
        if hasattr(item, "name"):
            name = item.name
            price = item.price
        else:
            name = item["name"]
            price = item["price"]
        subtotal += price
        lines.append(f"  - {name}: ${price:.2f}")

    tax = subtotal * tax_rate
    total = subtotal + tax

    result = "Budget Breakdown:\n"
    result += "\n".join(lines)
    result += f"\n\n  Subtotal: ${subtotal:.2f}"
    if tax_rate > 0:
        result += f"\n  Tax ({tax_rate:.0%}): ${tax:.2f}"
    result += f"\n  Total: ${total:.2f}"

    if budget_limit is not None:
        remaining = budget_limit - total
        if remaining >= 0:
            result += f"\n\n  Within budget (${remaining:.2f} remaining)"
        else:
            result += f"\n\n  OVER BUDGET by ${abs(remaining):.2f}"

    return result
