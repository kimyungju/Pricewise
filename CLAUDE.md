# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangGraph autonomous agent that searches for products via Tavily and returns structured Pydantic output (Receipt). Features conversation summarization and human-in-the-loop CLI approval.

## Tech Stack

Python 3.12+, uv (package manager), LangGraph, LangChain, OpenAI (gpt-4o), Tavily, Pydantic v2

## Commands

```bash
# Install dependencies
uv sync

# Run the agent
uv run python main.py

# Run all tests
uv run pytest -v

# Run a single test file
uv run pytest tests/test_schemas.py -v

# Run a single test
uv run pytest tests/test_schemas.py::test_receipt_defaults -v

# Run async tests (middleware tests)
uv run pytest tests/test_middleware.py -v  # requires pytest-asyncio
```

## Architecture

Uses `src/` layout: package is `src/aigent/`.

- **`agent.py`** — Builds the LangGraph agent via `create_react_agent`. This is the central orchestration point that wires together the model, tools, middleware hooks, and checkpointer.
- **`schemas.py`** — Pydantic models: `ProductQuery` (tool input) and `Receipt` (structured agent output via `response_format`).
- **`tools/search_product.py`** — `@tool`-decorated function using `TavilySearch`. Module-level `_tavily` client is overridable for testing.
- **`middleware/summarization.py`** — Factory `create_summarization_hook(model, max_messages)` returns an async `pre_model_hook` that compresses history via LLM when messages exceed threshold.
- **`middleware/human_approval.py`** — CLI `prompt_for_approval()` helper used in the execution loop when the graph hits `interrupt_before=["tools"]`.
- **`main.py`** — Async entrypoint. Handles the invoke → check interrupt → approve → resume loop with `InMemorySaver` checkpointing.

## Key LangGraph Patterns

- Agent is built with `create_react_agent` (not legacy `AgentExecutor`)
- Summarization uses `pre_model_hook` returning `{"llm_input_messages": ...}` to avoid mutating state
- HITL uses `interrupt_before=["tools"]` + `agent.aget_state()` to check `state.next` + resume with `ainvoke(None, config)`
- Structured output uses `response_format=Receipt`, accessed via `result["structured_response"]`
- Model initialized with `init_chat_model("gpt-4o", model_provider="openai")` for provider flexibility

## Environment

Requires `.env` file with `OPENAI_API_KEY` and `TAVILY_API_KEY`. See `.env.example`.

## Design Docs

Implementation plan and design rationale in `docs/plans/`.
