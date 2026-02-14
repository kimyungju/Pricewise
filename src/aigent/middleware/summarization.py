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
