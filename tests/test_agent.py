import os
from unittest.mock import patch
from langgraph.checkpoint.memory import InMemorySaver
from aigent.agent import build_agent


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "TAVILY_API_KEY": "tvly-test"})
def test_build_agent_returns_compiled_graph():
    agent = build_agent()
    assert hasattr(agent, "invoke")
    assert hasattr(agent, "ainvoke")
    assert hasattr(agent, "get_state")


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "TAVILY_API_KEY": "tvly-test"})
def test_build_agent_accepts_external_checkpointer():
    checkpointer = InMemorySaver()
    agent = build_agent(checkpointer=checkpointer)
    assert hasattr(agent, "ainvoke")
