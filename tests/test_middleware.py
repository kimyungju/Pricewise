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
