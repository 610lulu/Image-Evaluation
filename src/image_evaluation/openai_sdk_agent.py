import os
from typing import Literal

from image_evaluation.photo_agent import AgentResult, ToolCall
from image_evaluation.photo_tools import SYSTEM_INSTRUCTIONS, PhotoToolRuntime, compact_arguments


class OpenAIAgentsSDKPhotoAgent:
    """Photo agent implemented with OpenAI Agents SDK's managed agent/tool loop."""

    def __init__(self, model: str | None = None, max_turns: int = 8) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4")
        self.max_turns = max_turns

    def run(self, query: str) -> AgentResult:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("openai-sdk mode requires OPENAI_API_KEY")

        from agents import Agent, Runner, function_tool

        runtime = PhotoToolRuntime()
        observed_calls: list[ToolCall] = []

        @function_tool
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

        @function_tool
        def rank_photos(criterion: Literal["aesthetic_quality"], limit: int) -> str:
            """Rank current search results by aesthetic quality and return at most limit photos."""
            arguments = {"criterion": criterion, "limit": limit}
            observed_calls.append(ToolCall("rank_photos", arguments))
            return runtime.execute_json("rank_photos", arguments)

        agent = Agent(
            name="Photo Agent",
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            tools=[search_photos, rank_photos],
        )
        result = Runner.run_sync(agent, query, max_turns=self.max_turns)

        selected = runtime.selected or runtime.candidates
        answer = str(result.final_output or f"完成，共返回 {len(selected)} 张照片。")
        return AgentResult(answer=answer, tool_calls=observed_calls, selected_photo_ids=selected)
