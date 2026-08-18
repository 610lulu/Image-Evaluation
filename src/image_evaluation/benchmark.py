import argparse
import json
import time
from pathlib import Path

from image_evaluation.agent_factory import AGENT_MODES, build_agent
from image_evaluation.runner import run_suite


DEFAULT_BENCHMARK_AGENTS = ("responses", "openai-sdk", "langgraph")


def parse_agent_list(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [mode for mode in modes if mode not in AGENT_MODES]
    if invalid:
        raise ValueError(f"Unsupported agent mode(s): {invalid}")
    return modes


def run_benchmark(
    agent_modes: list[str],
    case_path: str = "data/cases.jsonl",
    model: str | None = None,
) -> dict:
    results = []

    for mode in agent_modes:
        started = time.perf_counter()
        try:
            agent = build_agent(mode, model=model)
            report = run_suite(case_path, agent=agent)
            elapsed = time.perf_counter() - started
            summary = report["summary"]
            results.append(
                {
                    "mode": mode,
                    "agent": summary["agent"],
                    "pass_rate": summary["pass_rate"],
                    "average_score": summary["average_score"],
                    "passed_cases": summary["passed_cases"],
                    "total_cases": summary["total_cases"],
                    "elapsed_seconds": round(elapsed, 3),
                    "error": None,
                }
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            results.append(
                {
                    "mode": mode,
                    "agent": None,
                    "pass_rate": 0.0,
                    "average_score": 0.0,
                    "passed_cases": 0,
                    "total_cases": 0,
                    "elapsed_seconds": round(elapsed, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    successful = [item for item in results if item["error"] is None]
    ranking = sorted(
        successful,
        key=lambda item: (item["pass_rate"], item["average_score"], -item["elapsed_seconds"]),
        reverse=True,
    )
    return {
        "model": model,
        "agent_modes": agent_modes,
        "results": results,
        "ranking": [item["mode"] for item in ranking],
    }


def write_benchmark_report(report: dict, output_dir: str | Path = "reports/benchmark") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Photo Agent Benchmark",
        "",
        f"- Model: {report['model'] or 'OPENAI_MODEL/default'}",
        f"- Ranking: {' > '.join(report['ranking']) if report['ranking'] else 'No successful runs'}",
        "",
        "| Mode | Agent | Pass Rate | Avg Score | Time (s) | Error |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in report["results"]:
        error = (item["error"] or "").replace("|", "/")
        lines.append(
            f"| {item['mode']} | {item['agent'] or '-'} | {item['pass_rate']:.2%} | "
            f"{item['average_score']:.3f} | {item['elapsed_seconds']:.3f} | {error} |"
        )
    (output / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple Photo Agent implementations")
    parser.add_argument(
        "--agents",
        default=",".join(DEFAULT_BENCHMARK_AGENTS),
        help="Comma-separated modes, e.g. responses,openai-sdk,langgraph",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--case-path", default="data/cases.jsonl")
    parser.add_argument("--output-dir", default="reports/benchmark")
    args = parser.parse_args()

    report = run_benchmark(parse_agent_list(args.agents), args.case_path, args.model)
    write_benchmark_report(report, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
