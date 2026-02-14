from aigent.api.streaming import format_sse_event


def test_format_token_event():
    result = format_sse_event("token", {"content": "Hello"})
    assert result == 'event: token\ndata: {"content": "Hello"}\n\n'


def test_format_approval_required_event():
    tool_calls = [{"id": "1", "name": "search_product", "args": {"query": "headphones"}}]
    result = format_sse_event("approval_required", {"tool_calls": tool_calls})
    assert "approval_required" in result
    assert "search_product" in result


def test_format_done_event():
    result = format_sse_event("done", {})
    assert result == 'event: done\ndata: {}\n\n'


def test_format_error_event():
    result = format_sse_event("error", {"message": "Something went wrong"})
    assert "error" in result
    assert "Something went wrong" in result
