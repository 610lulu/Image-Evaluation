import json
import os
from typing import Any

from image_evaluation.photo_agent import AgentResult, ToolCall


PHOTO_TOOLS = [
    {
        "type": "function",
        "name": "search_photos",
        "description": (
            "Search the user's photo library. Use only filters explicitly present in the user request. "
            "This must be called before rank_photos."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Normalized English city name, e.g. Tokyo."},
                "time": {"type": "string", "description": "Use last_year when the user says 去年."},
                "scene": {"type": "string", "description": "Use night for 夜景/晚上."},
                "object": {"type": "string", "description": "Object category, e.g. cherry_blossom or cat."},
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Excluded categories, e.g. [selfie].",
                },
            },
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "rank_photos",
        "description": (
            "Rank the current search results. Call only after search_photos when the user asks for best, "
            "精选, 挑出 or 最好看的 photos."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "criterion": {
                    "type": "string",
                    "enum": ["aesthetic_quality"],
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["criterion", "limit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


SYSTEM_INSTRUCTIONS = """You are a photo album agent used for evaluation.
Always satisfy photo-search requests through tools rather than claiming you searched without a tool call.
Call search_photos first. Only call rank_photos after search_photos when ranking/selection is requested.
Preserve explicit user constraints exactly. Normalize 东京=Tokyo, 北京=Beijing, 上海=Shanghai,
去年=last_year, 夜景/晚上=night, 樱花=cherry_blossom, 猫=cat, and excluded 自拍=selfie.
If the user requests 三张/3张 best photos, rank_photos.limit must be 3.
After the required tools finish, answer concisely in Chinese.
"""


class _PhotoToolRuntime:
    """Deterministic tool backend so the evaluation isolates agent behavior."""

    def __init__(self) -> None:
        self.candidates: list[str] = []
        self.selected: list[str] = []

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_photos":
            # Stable fake IDs keep the demo reproducible; a real product would query the gallery index here.
            self.candidates = [f"photo_{i:03d}" for i in range(1, 6)]
            self.selected = list(self.candidates)
            return {
                "photo_ids": self.candidates,
                "count": len(self.candidates),
                "applied_filters": arguments,
            }

        if name == "rank_photos":
            if not self.candidates:
                return {"error": "rank_photos requires search_photos first"}
            limit = int(arguments.get("limit", 5))
            self.selected = self.candidates[:limit]
            return {
                "photo_ids": self.selected,
                "count": len(self.selected),
                "criterion": arguments.get("criterion"),
            }

        return {"error": f"unknown tool: {name}"}


class LLMPhotoAgent:
    """Real LLM function-calling agent with deterministic photo tool stubs."""

    def __init__(
        self,
        model: str | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        max_steps: int = 6,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.6")
        self.max_steps = max_steps

        if client is not None:
            self.client = client
            return

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "LLM mode requires OPENAI_API_KEY. Export it first, or use --agent mock."
            )

        from openai import OpenAI

        self.client = OpenAI(api_key=resolved_key)

    @staticmethod
    def _function_calls(response: Any) -> list[Any]:
        return [item for item in response.output if getattr(item, "type", None) == "function_call"]

    def run(self, query: str) -> AgentResult:
        runtime = _PhotoToolRuntime()
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
                tool_result = runtime.execute(call.name, arguments)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(tool_result, ensure_ascii=False),
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
        return AgentResult(
            answer=answer,
            tool_calls=observed_calls,
            selected_photo_ids=selected,
        )
