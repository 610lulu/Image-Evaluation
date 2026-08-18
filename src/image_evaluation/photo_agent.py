from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCall]
    selected_photo_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "tool_calls": [asdict(call) for call in self.tool_calls],
            "selected_photo_ids": self.selected_photo_ids,
        }


class MockPhotoAgent:
    """Deterministic photo agent used as the zero-cost evaluation baseline."""

    def run(self, query: str) -> AgentResult:
        tool_calls: list[ToolCall] = []
        search_args: dict[str, Any] = {}

        if "东京" in query:
            search_args["location"] = "Tokyo"
        elif "北京" in query:
            search_args["location"] = "Beijing"
        elif "上海" in query:
            search_args["location"] = "Shanghai"

        if "去年" in query:
            search_args["time"] = "last_year"
        if "夜景" in query or "晚上" in query:
            search_args["scene"] = "night"
        if "樱花" in query:
            search_args["object"] = "cherry_blossom"
        if "猫" in query:
            search_args["object"] = "cat"
        if "自拍" in query and ("排除" in query or "不要" in query):
            search_args["exclude"] = ["selfie"]

        tool_calls.append(ToolCall("search_photos", search_args))

        limit = 3 if "三张" in query or "3张" in query else 5
        needs_ranking = any(word in query for word in ["最好看", "最好", "精选", "挑出"])
        if needs_ranking:
            tool_calls.append(
                ToolCall(
                    "rank_photos",
                    {
                        "criterion": "aesthetic_quality",
                        "limit": limit,
                    },
                )
            )

        count = limit if needs_ranking else min(limit, 5)
        selected = [f"photo_{i:03d}" for i in range(1, count + 1)]
        return AgentResult(
            answer=f"找到 {len(selected)} 张符合条件的照片。",
            tool_calls=tool_calls,
            selected_photo_ids=selected,
        )


# Backward-compatible name used by the original MVP.
PhotoAgent = MockPhotoAgent
