import uuid
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessageChunk

from aigent.api.streaming import format_sse_event

router = APIRouter()


class MessageRequest(BaseModel):
    content: str


class ApprovalRequest(BaseModel):
    approved: bool


def _get_session(request: Request, session_id: str) -> dict:
    """Look up a session or raise 404."""
    sessions = request.app.state.sessions
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


async def _stream_agent(agent, config, input_value):
    """Shared SSE generator used by both message and approve endpoints.

    Args:
        agent: The compiled LangGraph agent.
        config: The LangGraph runnable config with thread_id.
        input_value: The input to pass to agent.astream (dict for new message, None for resume).
    """
    try:
        async for mode, payload in agent.astream(
            input_value, config=config, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                message, _metadata = payload
                if isinstance(message, AIMessageChunk):
                    if message.content:
                        yield format_sse_event("token", {"content": message.content})
                    if message.tool_calls:
                        for tc in message.tool_calls:
                            yield format_sse_event("tool_call", {
                                "name": tc["name"],
                                "args": tc["args"],
                            })

        # After streaming completes, inspect the state
        state = await agent.aget_state(config)

        if state.next:
            # Agent is interrupted (waiting for tool approval)
            last_msg = state.values["messages"][-1]
            tool_calls = [
                {"name": tc["name"], "args": tc["args"]}
                for tc in last_msg.tool_calls
            ]
            yield format_sse_event("approval_required", {"tool_calls": tool_calls})
        else:
            # Check for structured Receipt
            structured = state.values.get("structured_response")
            if structured:
                yield format_sse_event("receipt", structured.model_dump())

        yield format_sse_event("done", {})

    except Exception as exc:
        yield format_sse_event("error", {"message": str(exc)})
        yield format_sse_event("done", {})


@router.post("/sessions")
async def create_session(request: Request):
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    request.app.state.sessions[session_id] = {
        "thread_id": session_id,
    }
    return {"session_id": session_id}


@router.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, body: MessageRequest, request: Request):
    """Send a user message and stream the agent's response via SSE."""
    session = _get_session(request, session_id)
    agent = request.app.state.agent
    config = {"configurable": {"thread_id": session["thread_id"]}}

    return StreamingResponse(
        _stream_agent(agent, config, {"messages": [("user", body.content)]}),
        media_type="text/event-stream",
    )


@router.post("/sessions/{session_id}/approve")
async def approve_tool(session_id: str, body: ApprovalRequest, request: Request):
    """Approve or deny pending tool calls, then stream the rest of the response."""
    session = _get_session(request, session_id)
    agent = request.app.state.agent
    config = {"configurable": {"thread_id": session["thread_id"]}}

    if not body.approved:
        async def denied_stream():
            yield format_sse_event("error", {"message": "Tool execution denied by user"})
            yield format_sse_event("done", {})

        return StreamingResponse(
            denied_stream(),
            media_type="text/event-stream",
        )

    return StreamingResponse(
        _stream_agent(agent, config, None),
        media_type="text/event-stream",
    )
