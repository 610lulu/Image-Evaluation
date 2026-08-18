from typing import Any

from image_evaluation.llm_agent import LLMPhotoAgent
from image_evaluation.photo_agent import MockPhotoAgent


def build_agent(mode: str = "mock", model: str | None = None, client: Any | None = None):
    if mode == "mock":
        return MockPhotoAgent()
    if mode == "llm":
        return LLMPhotoAgent(model=model, client=client)
    raise ValueError(f"Unsupported agent mode: {mode}")
