import json
from types import SimpleNamespace

from image_evaluation.agent_factory import build_agent
from image_evaluation.benchmark import parse_agent_list, run_benchmark
from image_evaluation.langgraph_agent import LangGraphPhotoAgent
from image_evaluation.llm_agent import LLMPhotoAgent, ResponsesPhotoAgent
from image_evaluation.openai_sdk_agent import OpenAIAgentsSDKPhotoAgent
from image_evaluation.photo_agent import MockPhotoAgent
from image_evaluation.runner import run_suite


def test_mvp_suite_all_cases_pass():
    report = run_suite()
    assert report["summary"]["total_cases"] >= 5
    assert report["summary"]["pass_rate"] == 1.0
    assert report["summary"]["agent"] == "MockPhotoAgent"


def test_scores_are_bounded():
    report = run_suite()
    for case in report["cases"]:
        assert 0.0 <= case["overall_score"] <= 1.0


def test_agent_factory_supports_framework_adapters_without_network():
    assert isinstance(build_agent("mock"), MockPhotoAgent)
    assert isinstance(build_agent("openai-sdk", model="test-model"), OpenAIAgentsSDKPhotoAgent)
    assert isinstance(build_agent("langgraph", model="test-model"), LangGraphPhotoAgent)


class _FakeResponses:
    def __init__(self):
        self.calls = []
        self._responses = [
            SimpleNamespace(
                id="resp_1",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="search_photos",
                        arguments=json.dumps(
                            {
                                "location": "Tokyo",
                                "time": "last_year",
                                "scene": "night",
                                "exclude": ["selfie"],
                            }
                        ),
                        call_id="call_1",
                    )
                ],
                output_text="",
            ),
            SimpleNamespace(
                id="resp_2",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="rank_photos",
                        arguments=json.dumps({"criterion": "aesthetic_quality", "limit": 3}),
                        call_id="call_2",
                    )
                ],
                output_text="",
            ),
            SimpleNamespace(id="resp_3", output=[], output_text="已选出三张照片。"),
        ]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


def test_responses_agent_tool_loop_can_be_evaluated_without_network():
    agent = LLMPhotoAgent(model="test-model", client=_FakeClient())
    assert isinstance(agent, ResponsesPhotoAgent)
    result = agent.run("找出去年在东京拍的夜景照片，排除自拍，然后选出最好看的三张。")

    assert [call.name for call in result.tool_calls] == ["search_photos", "rank_photos"]
    assert result.tool_calls[0].arguments["location"] == "Tokyo"
    assert result.tool_calls[1].arguments["limit"] == 3
    assert len(result.selected_photo_ids) == 3


def test_benchmark_can_compare_mock_offline():
    report = run_benchmark(["mock"], case_path="data/cases.jsonl")
    assert report["ranking"] == ["mock"]
    assert report["results"][0]["pass_rate"] == 1.0
    assert report["results"][0]["error"] is None


def test_parse_agent_list():
    assert parse_agent_list("responses, openai-sdk,langgraph") == [
        "responses",
        "openai-sdk",
        "langgraph",
    ]
