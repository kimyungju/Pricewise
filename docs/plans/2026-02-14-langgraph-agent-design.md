# LangGraph Autonomous Agent Design

## Overview

A production-ready autonomous agent built with LangGraph's `create_react_agent`, featuring structured output, conversation summarization, and human-in-the-loop approval. The agent searches for products via Tavily and returns a structured Receipt.

## Architecture

**Approach:** Modular package with clear separation of concerns.

**Stack:** LangGraph + LangChain + OpenAI (gpt-4o) + Tavily + Pydantic + uv

## Project Structure

```
aigent/
├── pyproject.toml
├── .env / .env.example
├── src/aigent/
│   ├── __init__.py
│   ├── agent.py              # Graph construction
│   ├── schemas.py            # Pydantic models
│   ├── tools/
│   │   ├── __init__.py
│   │   └── search_product.py # @tool with Tavily
│   └── middleware/
│       ├── __init__.py
│       ├── summarization.py  # Message compression
│       └── human_approval.py # CLI interrupt
├── main.py                   # Async entrypoint
```

## Data Models (`schemas.py`)

- **ProductQuery**: `query: str`, `max_results: int = 3` — input validation for SearchProduct tool
- **Receipt**: `product_name: str`, `price: float`, `currency: str = "USD"` — structured final output

## Tool: SearchProduct (`tools/search_product.py`)

- Decorated with `@tool(args_schema=ProductQuery)`
- Uses `TavilySearchResults` internally for real web search
- Returns formatted product results as string for LLM reasoning

## Middleware

### Summarization (`middleware/summarization.py`)

- Implemented as a `state_modifier` function passed to `create_react_agent`
- Triggers when message count exceeds 5
- Summarizes older messages via LLM, keeps last 2 messages intact
- Replaces history with a single SystemMessage summary

### Human-in-the-Loop (`middleware/human_approval.py`)

- Uses LangGraph's `interrupt()` mechanism
- Pauses before SearchProduct execution
- CLI prompt: "Agent wants to search for '{query}'. Approve? [y/n]"
- On approval: resumes with `Command(resume=True)`
- On rejection: resumes with denial message to agent

## Agent Graph (`agent.py`)

```python
def build_agent():
    model = init_chat_model("gpt-4o")
    checkpointer = InMemorySaver()
    agent = create_react_agent(
        model=model,
        tools=[search_product],
        checkpointer=checkpointer,
        state_modifier=summarize_messages,
        response_format=Receipt,
    )
    return agent
```

HITL interrupt is injected by wrapping the tool-calling node.

## Execution (`main.py`)

- Async `main()` function
- Thread config with `InMemorySaver` for checkpointing
- Demonstrates invoke, interrupt/resume for HITL, and structured Receipt output

## Dependencies

`langchain`, `langchain-openai`, `langchain-community`, `langgraph`, `langchain-tavily`, `pydantic`, `python-dotenv`

## Design Rationale

LangGraph's functional `create_react_agent` over old `AgentExecutor`:
- Declarative graph vs. opaque execution loop
- Native streaming, checkpointing, and interrupt support
- Composable nodes instead of rigid chain structure
