from aigent.tools.calculate_budget import calculate_budget


def test_calculate_budget_basic():
    result = calculate_budget.invoke({
        "items": [
            {"name": "Headphones", "price": 59.99},
            {"name": "Case", "price": 12.99},
        ],
    })
    assert "72.98" in result
    assert "Headphones" in result


def test_calculate_budget_with_tax():
    result = calculate_budget.invoke({
        "items": [{"name": "Speaker", "price": 100.00}],
        "tax_rate": 0.08,
    })
    assert "108.00" in result
    assert "8%" in result


def test_calculate_budget_within_budget():
    result = calculate_budget.invoke({
        "items": [{"name": "Earbuds", "price": 49.99}],
        "budget_limit": 100.0,
    })
    assert "Within budget" in result


def test_calculate_budget_over_budget():
    result = calculate_budget.invoke({
        "items": [{"name": "Earbuds", "price": 149.99}],
        "budget_limit": 100.0,
    })
    assert "OVER BUDGET" in result


def test_calculate_budget_empty():
    result = calculate_budget.invoke({"items": []})
    assert "No items" in result
