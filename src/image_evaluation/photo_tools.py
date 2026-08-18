import json
from typing import Any


SYSTEM_INSTRUCTIONS = """You are a photo album agent used for evaluation.
Always satisfy photo-search requests through tools rather than claiming you searched without a tool call.
Call search_photos first. Only call rank_photos after search_photos when ranking/selection is requested.
Preserve explicit user constraints exactly. Normalize 东京=Tokyo, 北京=Beijing, 上海=Shanghai,
去年=last_year, 夜景/晚上=night, 樱花=cherry_blossom, 猫=cat, and excluded 自拍=selfie.
If the user requests 三张/3张 best photos, rank_photos.limit must be 3.
After the required tools finish, answer concisely in Chinese.
"""


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
                "criterion": {"type": "string", "enum": ["aesthetic_quality"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["criterion", "limit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class PhotoToolRuntime:
    """Deterministic tool backend so framework comparisons isolate agent behavior."""

    def __init__(self) -> None:
        self.candidates: list[str] = []
        self.selected: list[str] = []

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_photos":
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

    def execute_json(self, name: str, arguments: dict[str, Any]) -> str:
        return json.dumps(self.execute(name, arguments), ensure_ascii=False)


def compact_arguments(**kwargs: Any) -> dict[str, Any]:
    """Drop None values so optional tool fields compare cleanly with the golden cases."""
    return {key: value for key, value in kwargs.items() if value is not None}
