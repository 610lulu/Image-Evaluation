from dataclasses import dataclass
from typing import Any


@dataclass
class MetricResult:
    name: str
    score: float
    passed: bool
    reason: str


def _actual_tool_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {call["name"]: call.get("arguments", {}) for call in result.get("tool_calls", [])}


def evaluate_task_completion(case: dict[str, Any], result: dict[str, Any]) -> MetricResult:
    expected_count = case.get("expected_photo_count")
    actual_count = len(result.get("selected_photo_ids", []))
    passed = expected_count is None or expected_count == actual_count
    score = 1.0 if passed else 0.0
    reason = f"expected_photo_count={expected_count}, actual={actual_count}"
    return MetricResult("task_completion", score, passed, reason)


def evaluate_tool_correctness(case: dict[str, Any], result: dict[str, Any]) -> MetricResult:
    expected = case.get("expected_tools", [])
    actual = [call["name"] for call in result.get("tool_calls", [])]
    passed = actual == expected
    if not expected:
        score = 1.0 if not actual else 0.0
    else:
        matched = sum(1 for i, tool in enumerate(expected) if i < len(actual) and actual[i] == tool)
        score = matched / len(expected)
    return MetricResult(
        "tool_correctness",
        round(score, 4),
        passed,
        f"expected={expected}, actual={actual}",
    )


def evaluate_argument_correctness(case: dict[str, Any], result: dict[str, Any]) -> MetricResult:
    expected_calls = case.get("expected_arguments", {})
    if not expected_calls:
        return MetricResult("argument_correctness", 1.0, True, "no expected arguments")

    actual_map = _actual_tool_map(result)
    total = 0
    matched = 0
    mismatches: list[str] = []

    for tool_name, expected_args in expected_calls.items():
        actual_args = actual_map.get(tool_name, {})
        for key, expected_value in expected_args.items():
            total += 1
            actual_value = actual_args.get(key)
            if actual_value == expected_value:
                matched += 1
            else:
                mismatches.append(
                    f"{tool_name}.{key}: expected={expected_value!r}, actual={actual_value!r}"
                )

    score = matched / total if total else 1.0
    passed = matched == total
    reason = "all arguments matched" if passed else "; ".join(mismatches)
    return MetricResult("argument_correctness", round(score, 4), passed, reason)


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    metrics = [
        evaluate_task_completion(case, result),
        evaluate_tool_correctness(case, result),
        evaluate_argument_correctness(case, result),
    ]
    overall = sum(metric.score for metric in metrics) / len(metrics)
    return {
        "case_id": case["id"],
        "query": case["query"],
        "tags": case.get("tags", []),
        "passed": all(metric.passed for metric in metrics),
        "overall_score": round(overall, 4),
        "metrics": [metric.__dict__ for metric in metrics],
        "result": result,
    }
