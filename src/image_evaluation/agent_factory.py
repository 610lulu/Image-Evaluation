from typing import Any

from image_evaluation.llm_agent import ResponsesPhotoAgent
from image_evaluation.photo_agent import MockPhotoAgent


AGENT_MODES = ("mock", "responses", "openai-sdk", "langgraph", "llm")


def build_agent(mode: str = "mock", model: str | None = None, client: Any | None = None):
    if mode == "mock":
        return MockPhotoAgent()
    if mode in {"responses", "llm"}:
        return ResponsesPhotoAgent(model=model, client=client)
    if mode == "openai-sdk":
        from image_evaluation.openai_sdk_agent import OpenAIAgentsSDKPhotoAgent

        return OpenAIAgentsSDKPhotoAgent(model=model)
    if mode == "langgraph":
        from image_evaluation.langgraph_agent import LangGraphPhotoAgent

        return LangGraphPhotoAgent(model=model)
    raise ValueError(f"Unsupported agent mode: {mode}. Expected one of {AGENT_MODES}")
