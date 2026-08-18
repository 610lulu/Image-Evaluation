import json
import os
from typing import Any

from image_evaluation.photo_agent import AgentResult, ToolCall
from image_evaluation.photo_tools import PHOTO_TOOLS, SYSTEM_INSTRUCTIONS, PhotoToolRuntime


class ResponsesPhotoAgent:
    """Low-level Responses API agent where this project owns the tool loop."""

    def __init__(
        self,
        model: str | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        max_steps: int = 6,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4")
        self.max_steps = max_steps

        if client is not None:
            self.client = client
            return

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "Responses mode requires OPENAI_API_KEY. Export it first, or use --agent mock."
            )

        from openai import OpenAI

        self.client = OpenAI(api_key=resolved_key)

    @staticmethod
    def _function_calls(response: Any) -> list[Any]:
        return [item for item in response.output if getattr(item, "type", None) == "function_call"]

    def run(self, query: str) -> AgentResult:
        runtime = PhotoToolRuntime()
        observed_calls: list[ToolCall] = []

        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=query,
            tools=PHOTO_TOOLS,
            tool_choice="required",
            parallel_tool_calls=False,
        )

        for _ in range(self.max_steps):
            function_calls = self._function_calls(response)
            if not function_calls:
                break

            tool_outputs: list[dict[str, Any]] = []
            for call in function_calls:
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError:
                    arguments = {"_invalid_json": call.arguments}

                observed_calls.append(ToolCall(name=call.name, arguments=arguments))
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": runtime.execute_json(call.name, arguments),
                    }
                )

            response = self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_INSTRUCTIONS,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=PHOTO_TOOLS,
                tool_choice="auto",
                parallel_tool_calls=False,
            )
        else:
            raise RuntimeError(f"Agent exceeded max_steps={self.max_steps}")

        selected = runtime.selected or runtime.candidates
        answer = getattr(response, "output_text", "") or f"完成，共返回 {len(selected)} 张照片。"
        return AgentResult(answer=answer, tool_calls=observed_calls, selected_photo_ids=selected)


LLMPhotoAgent = ResponsesPhotoAgent
