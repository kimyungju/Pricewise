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
from aigent.tools import search_product, compare_prices, get_reviews, calculate_budget
from aigent.middleware.summarization import create_summarization_hook


def build_agent(checkpointer=None):
    """Build and return the compiled agent graph.

    The agent:
      1. Uses gpt-4o via init_chat_model (provider-agnostic initialization)
      2. Has a SearchProduct tool backed by Tavily
      3. Summarizes conversation history after 5 messages (pre_model_hook)
      4. Pauses for human approval before any tool execution (interrupt_before)
      5. Returns a structured Receipt as its final output (response_format)
    """
    model = init_chat_model("gpt-4o", model_provider="openai")
    if checkpointer is None:
        checkpointer = InMemorySaver()

    # Summarization hook: compresses history when messages exceed threshold
    summarization_hook = create_summarization_hook(model, max_messages=5)

    agent = create_react_agent(
        model=model,
        tools=[search_product, compare_prices, get_reviews, calculate_budget],
        checkpointer=checkpointer,
        pre_model_hook=summarization_hook,
        interrupt_before=["tools"],  # HITL: pause before tool execution
        response_format=Receipt,
    )

    return agent
