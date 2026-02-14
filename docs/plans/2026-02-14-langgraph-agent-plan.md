# LangGraph Autonomous Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-ready LangGraph autonomous agent with structured output, conversation summarization, and human-in-the-loop approval.

**Architecture:** Modular Python package using LangGraph's `create_react_agent` with `pre_model_hook` for summarization, `interrupt_before` for HITL, and `response_format` for structured Receipt output. Tavily provides real product search.

**Tech Stack:** Python 3.12+, uv, LangGraph, LangChain, OpenAI (gpt-4o), Tavily, Pydantic v2, pytest

---

### Task 1: Project scaffolding with uv

**Files:**
- Create: `pyproject.toml` (via uv)
- Create: `src/aigent/__init__.py` (via uv)
- Create: `.env.example`
- Create: `.gitignore`

**Step 1: Initialize uv package**

Run: `uv init --package`
Expected: Creates `pyproject.toml` and `src/aigent/__init__.py`

**Step 2: Add production dependencies**

Run: `uv add langchain langchain-openai langchain-community langgraph langchain-tavily pydantic python-dotenv`
Expected: Dependencies added to `pyproject.toml`, `uv.lock` created

**Step 3: Add dev dependencies**

Run: `uv add --dev pytest pytest-asyncio`
Expected: Dev dependencies added under `[tool.uv]` or `[dependency-groups]`

**Step 4: Create .env.example**

```
OPENAI_API_KEY=sk-your-openai-api-key-here
TAVILY_API_KEY=tvly-your-tavily-api-key-here
```

**Step 5: Create .gitignore**

```
.venv/
__pycache__/
*.pyc
.env
uv.lock
*.egg-info/
dist/
```

**Step 6: Create module directories**

Run:
```bash
mkdir -p src/aigent/tools src/aigent/middleware tests
touch src/aigent/tools/__init__.py src/aigent/middleware/__init__.py tests/__init__.py
```

**Step 7: Initialize git and commit**

```bash
git init
git add pyproject.toml .env.example .gitignore src/ tests/
git commit -m "chore: scaffold aigent project with uv"
```

---

### Task 2: Pydantic schemas

**Files:**
- Create: `src/aigent/schemas.py`
- Create: `tests/test_schemas.py`

**Step 1: Write failing tests for schemas**

```python
# tests/test_schemas.py
from aigent.schemas import ProductQuery, Receipt


def test_product_query_defaults():
    q = ProductQuery(query="headphones")
    assert q.query == "headphones"
    assert q.max_results == 3


def test_product_query_custom_max():
    q = ProductQuery(query="laptop", max_results=5)
    assert q.max_results == 5


def test_receipt_defaults():
    r = Receipt(product_name="AirPods", price=199.99)
    assert r.product_name == "AirPods"
    assert r.price == 199.99
    assert r.currency == "USD"


def test_receipt_custom_currency():
    r = Receipt(product_name="Sony WH-1000XM5", price=349.99, currency="EUR")
    assert r.currency == "EUR"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigent.schemas'`

**Step 3: Implement schemas**

```python
# src/aigent/schemas.py
from pydantic import BaseModel, Field


class ProductQuery(BaseModel):
    """Input schema for the SearchProduct tool."""
    query: str = Field(description="The product search term")
    max_results: int = Field(default=3, description="Maximum number of results to return")


class Receipt(BaseModel):
    """Structured output schema for the agent's final response."""
    product_name: str = Field(description="Name of the recommended product")
    price: float = Field(description="Price of the product")
    currency: str = Field(default="USD", description="Currency code")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/aigent/schemas.py tests/test_schemas.py
git commit -m "feat: add ProductQuery and Receipt Pydantic schemas"
```

---

### Task 3: SearchProduct tool

**Files:**
- Create: `src/aigent/tools/search_product.py`
- Create: `tests/test_tools.py`
- Modify: `src/aigent/tools/__init__.py`

**Step 1: Write failing test for the tool**

```python
# tests/test_tools.py
from unittest.mock import patch, MagicMock
from aigent.tools.search_product import search_product


def test_search_product_returns_formatted_string():
    mock_results = [
        {"url": "https://example.com/headphones", "content": "Best wireless headphones - Sony WH-1000XM5 for $349"},
        {"url": "https://example.com/buds", "content": "AirPods Pro 2 - $249 great noise cancellation"},
    ]

    with patch("aigent.tools.search_product.TavilySearch") as MockTavily:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_results
        MockTavily.return_value = mock_instance

        # Re-import to trigger module-level TavilySearch init
        import importlib
        import aigent.tools.search_product as mod
        mod._tavily = mock_instance

        result = search_product.invoke({"query": "wireless headphones", "max_results": 2})
        assert isinstance(result, str)
        assert "Sony" in result or "headphones" in result
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement the tool**

```python
# src/aigent/tools/search_product.py
import os
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from aigent.schemas import ProductQuery

# Module-level Tavily client (overridable for testing)
_tavily = TavilySearch(max_results=5, topic="general")


@tool(args_schema=ProductQuery)
def search_product(query: str, max_results: int = 3) -> str:
    """Search for a product online using Tavily and return formatted results."""
    results = _tavily.invoke({"query": query, "max_results": max_results})

    if not results:
        return "No products found for this query."

    formatted = []
    for i, r in enumerate(results[:max_results], 1):
        url = r.get("url", "N/A")
        content = r.get("content", "No description")
        formatted.append(f"{i}. {content}\n   URL: {url}")

    return "\n\n".join(formatted)
```

**Step 4: Update tools __init__.py**

```python
# src/aigent/tools/__init__.py
from aigent.tools.search_product import search_product

__all__ = ["search_product"]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_tools.py -v`
Expected: 1 passed

**Step 6: Commit**

```bash
git add src/aigent/tools/ tests/test_tools.py
git commit -m "feat: add SearchProduct tool with Tavily integration"
```

---

### Task 4: Summarization middleware

**Files:**
- Create: `src/aigent/middleware/summarization.py`
- Create: `tests/test_middleware.py`
- Modify: `src/aigent/middleware/__init__.py`

**Step 1: Write failing test for summarization hook**

```python
# tests/test_middleware.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from aigent.middleware.summarization import create_summarization_hook


@pytest.mark.asyncio
async def test_no_summarization_under_threshold():
    """Messages under threshold are passed through unchanged."""
    mock_model = MagicMock()
    hook = create_summarization_hook(mock_model, max_messages=5)

    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
    ]
    result = await hook({"messages": messages})
    assert result["llm_input_messages"] == messages
    mock_model.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_summarization_over_threshold():
    """Messages over threshold trigger summarization."""
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = AIMessage(content="User asked about headphones. Agent recommended Sony.")

    hook = create_summarization_hook(mock_model, max_messages=5)

    messages = [
        HumanMessage(content="msg1"),
        AIMessage(content="reply1"),
        HumanMessage(content="msg2"),
        AIMessage(content="reply2"),
        HumanMessage(content="msg3"),
        AIMessage(content="reply3"),
        HumanMessage(content="msg4"),  # 7 messages > 5
    ]
    result = await hook({"messages": messages})

    # Should have a summary SystemMessage + last 2 messages
    llm_messages = result["llm_input_messages"]
    assert isinstance(llm_messages[0], SystemMessage)
    assert "Summary" in llm_messages[0].content
    assert len(llm_messages) == 3  # summary + last 2
    mock_model.ainvoke.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_middleware.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement the summarization hook**

```python
# src/aigent/middleware/summarization.py
from langchain_core.messages import SystemMessage


def create_summarization_hook(model, max_messages: int = 5):
    """Create a pre_model_hook that summarizes conversation history.

    Uses the provided model to compress older messages into a summary,
    keeping only the last 2 messages for immediate context.
    """

    async def summarize_messages(state: dict) -> dict:
        messages = state["messages"]

        if len(messages) <= max_messages:
            return {"llm_input_messages": messages}

        # Split: older messages to summarize, recent to keep
        messages_to_summarize = messages[:-2]
        recent_messages = messages[-2:]

        # Build summarization prompt
        summary_prompt = [
            SystemMessage(
                content=(
                    "Summarize the following conversation concisely. "
                    "Preserve key facts, decisions, and product details mentioned."
                )
            ),
            *messages_to_summarize,
        ]

        summary_response = await model.ainvoke(summary_prompt)

        summarized_messages = [
            SystemMessage(content=f"Summary of earlier conversation:\n{summary_response.content}"),
            *recent_messages,
        ]

        return {"llm_input_messages": summarized_messages}

    return summarize_messages
```

**Step 4: Update middleware __init__.py**

```python
# src/aigent/middleware/__init__.py
from aigent.middleware.summarization import create_summarization_hook

__all__ = ["create_summarization_hook"]
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_middleware.py -v`
Expected: 2 passed

**Step 6: Commit**

```bash
git add src/aigent/middleware/ tests/test_middleware.py
git commit -m "feat: add summarization pre_model_hook middleware"
```

---

### Task 5: Human approval helper

**Files:**
- Create: `src/aigent/middleware/human_approval.py`
- Modify: `src/aigent/middleware/__init__.py`

**Step 1: Implement the human approval helper**

This module provides a helper used in the execution loop (main.py) to prompt
the user when the agent is interrupted before the tools node.

```python
# src/aigent/middleware/human_approval.py


def prompt_for_approval(tool_calls: list) -> bool:
    """Display pending tool calls and prompt user for CLI approval.

    Args:
        tool_calls: List of tool call dicts from the AI message.

    Returns:
        True if user approves, False otherwise.
    """
    print("\n--- Human Approval Required ---")
    for tc in tool_calls:
        print(f"  Tool:  {tc['name']}")
        print(f"  Args:  {tc['args']}")
    print("-------------------------------")

    while True:
        answer = input("Approve execution? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")
```

**Step 2: Update middleware __init__.py**

```python
# src/aigent/middleware/__init__.py
from aigent.middleware.summarization import create_summarization_hook
from aigent.middleware.human_approval import prompt_for_approval

__all__ = ["create_summarization_hook", "prompt_for_approval"]
```

**Step 3: Commit**

```bash
git add src/aigent/middleware/
git commit -m "feat: add human-in-the-loop CLI approval helper"
```

---

### Task 6: Agent graph construction

**Files:**
- Create: `src/aigent/agent.py`
- Create: `tests/test_agent.py`

**Step 1: Write failing test for agent construction**

```python
# tests/test_agent.py
import os
from unittest.mock import patch
from aigent.agent import build_agent


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "TAVILY_API_KEY": "tvly-test"})
def test_build_agent_returns_compiled_graph():
    agent = build_agent()
    # Verify it's a compiled LangGraph with expected nodes
    assert hasattr(agent, "invoke")
    assert hasattr(agent, "ainvoke")
    assert hasattr(agent, "get_state")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement agent.py**

```python
# src/aigent/agent.py
"""Agent graph construction.

Uses LangGraph's create_react_agent — the functional, graph-based successor
to the old class-based AgentExecutor. Key advantages:
  - Declarative graph: nodes and edges are explicit, not hidden in a loop
  - Native checkpointing: state persists across invocations via configurable thread IDs
  - Built-in interrupt: human-in-the-loop via interrupt_before, no custom chains needed
  - Composable hooks: pre_model_hook for message management, response_format for output
"""
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from aigent.schemas import Receipt
from aigent.tools import search_product
from aigent.middleware.summarization import create_summarization_hook


def build_agent():
    """Build and return the compiled agent graph.

    The agent:
      1. Uses gpt-4o via init_chat_model (provider-agnostic initialization)
      2. Has a SearchProduct tool backed by Tavily
      3. Summarizes conversation history after 5 messages (pre_model_hook)
      4. Pauses for human approval before any tool execution (interrupt_before)
      5. Returns a structured Receipt as its final output (response_format)
    """
    model = init_chat_model("gpt-4o", model_provider="openai")
    checkpointer = InMemorySaver()

    # Summarization hook: compresses history when messages exceed threshold
    summarization_hook = create_summarization_hook(model, max_messages=5)

    agent = create_react_agent(
        model=model,
        tools=[search_product],
        checkpointer=checkpointer,
        pre_model_hook=summarization_hook,
        interrupt_before=["tools"],  # HITL: pause before tool execution
        response_format=Receipt,
    )

    return agent
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add src/aigent/agent.py tests/test_agent.py
git commit -m "feat: build agent graph with summarization, HITL, and structured output"
```

---

### Task 7: Async main entrypoint

**Files:**
- Create: `main.py`

**Step 1: Implement main.py**

```python
# main.py
"""Async entrypoint demonstrating the LangGraph autonomous agent.

This uses the functional create_react_agent approach instead of the legacy
AgentExecutor. The key difference: instead of an opaque while-loop that
calls the LLM and tools in sequence, we have a compiled state graph with
explicit nodes, edges, and checkpointing — making the execution fully
inspectable, interruptible, and resumable.
"""
import asyncio
import os

from dotenv import load_dotenv

from aigent.agent import build_agent
from aigent.middleware.human_approval import prompt_for_approval


async def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in .env")
        return
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not set in .env")
        return

    agent = build_agent()

    # Thread config enables checkpointing — the agent remembers state
    # across invocations within the same thread, which powers the
    # interrupt/resume flow for human-in-the-loop approval.
    config = {"configurable": {"thread_id": "demo-session-1"}}

    user_query = "Find me the best wireless headphones under $100"
    print(f"\nUser: {user_query}\n")

    # First invocation: the agent reasons and decides to call SearchProduct.
    # Because interrupt_before=["tools"], execution pauses before the tool runs.
    result = await agent.ainvoke(
        {"messages": [("user", user_query)]},
        config=config,
    )

    # Check if the agent is at an interrupt point (pending tool execution)
    state = await agent.aget_state(config)

    while state.next:
        # The agent wants to call a tool — show it and ask for approval
        last_msg = state.values["messages"][-1]

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            approved = prompt_for_approval(last_msg.tool_calls)

            if not approved:
                print("\nTool execution denied. Ending session.")
                return

        # Resume: passing None continues from the checkpoint
        result = await agent.ainvoke(None, config=config)
        state = await agent.aget_state(config)

    # Extract the structured Receipt output
    receipt = result.get("structured_response")

    if receipt:
        print("\n=== Final Receipt ===")
        print(f"  Product: {receipt.product_name}")
        print(f"  Price:   {receipt.price} {receipt.currency}")
        print("=====================")
    else:
        # Fallback: print the last agent message
        last_msg = result["messages"][-1]
        print(f"\nAgent: {last_msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Verify it runs (smoke test — requires real API keys)**

Run: `uv run python main.py`
Expected: Agent asks for approval, searches, returns a Receipt

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add async main entrypoint with HITL demo loop"
```

---

### Task 8: Create .env and final integration

**Files:**
- Create: `.env` (from `.env.example`, user fills in keys)

**Step 1: Copy .env.example to .env**

Run: `cp .env.example .env`
Expected: `.env` file created (user must fill in real keys)

**Step 2: Run full integration test**

Run: `uv run python main.py`
Expected:
1. Agent receives query about wireless headphones
2. Agent decides to call SearchProduct
3. Execution pauses, CLI shows "Approve execution? [y/n]"
4. On "y", Tavily searches, agent processes results
5. Agent returns a structured Receipt with product_name, price, currency

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: finalize project setup and integration"
```

---

## Summary of files created

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project config and dependencies |
| `.env.example` | API key template |
| `.gitignore` | Git ignore patterns |
| `src/aigent/__init__.py` | Package init |
| `src/aigent/schemas.py` | ProductQuery + Receipt models |
| `src/aigent/agent.py` | Agent graph construction |
| `src/aigent/tools/__init__.py` | Tools package |
| `src/aigent/tools/search_product.py` | Tavily-backed search tool |
| `src/aigent/middleware/__init__.py` | Middleware package |
| `src/aigent/middleware/summarization.py` | Message summarization hook |
| `src/aigent/middleware/human_approval.py` | CLI approval helper |
| `main.py` | Async entrypoint |
| `tests/test_schemas.py` | Schema unit tests |
| `tests/test_tools.py` | Tool unit tests |
| `tests/test_middleware.py` | Middleware unit tests |
| `tests/test_agent.py` | Agent construction test |
