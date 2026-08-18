import os
from typing import Literal

from image_evaluation.photo_agent import AgentResult, ToolCall
from image_evaluation.photo_tools import SYSTEM_INSTRUCTIONS, PhotoToolRuntime, compact_arguments


def _langchain_model_name(model: str) -> str:
    return model if ":" in model else f"openai:{model}"


class LangGraphPhotoAgent:
    """LangChain create_agent implementation; the runtime is backed by LangGraph."""

    def __init__(self, model: str | None = None, recursion_limit: int = 20) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4")
        self.recursion_limit = recursion_limit

    def run(self, query: str) -> AgentResult:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("langgraph mode requires OPENAI_API_KEY")

        from langchain.agents import create_agent
        from langchain.tools import tool

        runtime = PhotoToolRuntime()
        observed_calls: list[ToolCall] = []

        @tool
        def search_photos(
            location: str | None = None,
            time: str | None = None,
            scene: str | None = None,
            object: str | None = None,
            exclude: list[str] | None = None,
        ) -> str:
            """Search the user's photo library using only filters explicitly requested by the user."""
            arguments = compact_arguments(
                location=location,
                time=time,
                scene=scene,
                object=object,
                exclude=exclude,
            )
            observed_calls.append(ToolCall("search_photos", arguments))
            return runtime.execute_json("search_photos", arguments)

        @tool
        def rank_photos(criterion: Literal["aesthetic_quality"], limit: int) -> str:
            """Rank current search results by aesthetic quality and return at most limit photos."""
            arguments = {"criterion": criterion, "limit": limit}
            observed_calls.append(ToolCall("rank_photos", arguments))
            return runtime.execute_json("rank_photos", arguments)

        agent = create_agent(
            model=_langchain_model_name(self.model),
            tools=[search_photos, rank_photos],
            system_prompt=SYSTEM_INSTRUCTIONS,
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": self.recursion_limit},
        )

        selected = runtime.selected or runtime.candidates
        last_message = result["messages"][-1]
        answer = getattr(last_message, "text", "") or getattr(last_message, "content", "")
        if not isinstance(answer, str):
            answer = str(answer)
        if not answer:
            answer = f"完成，共返回 {len(selected)} 张照片。"
        return AgentResult(answer=answer, tool_calls=observed_calls, selected_photo_ids=selected)
