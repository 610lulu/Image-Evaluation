import json
from pathlib import Path

from image_evaluation.evaluators import evaluate_case
from image_evaluation.photo_agent import PhotoAgent


def load_cases(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_suite(case_path: str | Path = "data/cases.jsonl") -> dict:
    agent = PhotoAgent()
    cases = load_cases(case_path)
    details = []

    for case in cases:
        result = agent.run(case["query"]).to_dict()
        details.append(evaluate_case(case, result))

    total = len(details)
    passed = sum(1 for item in details if item["passed"])
    avg_score = sum(item["overall_score"] for item in details) / total if total else 0.0

    return {
        "summary": {
            "total_cases": total,
            "passed_cases": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "average_score": round(avg_score, 4),
        },
        "cases": details,
    }


def write_reports(report: dict, output_dir: str | Path = "reports") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = report["summary"]
    lines = [
        "# Photo Agent Evaluation Report",
        "",
        f"- Total cases: {summary['total_cases']}",
        f"- Passed cases: {summary['passed_cases']}",
        f"- Pass rate: {summary['pass_rate']:.2%}",
        f"- Average score: {summary['average_score']:.3f}",
        "",
        "## Case Results",
        "",
        "| Case | Passed | Score | Tags |",
        "|---|---:|---:|---|",
    ]
    for item in report["cases"]:
        lines.append(
            f"| {item['case_id']} | {'PASS' if item['passed'] else 'FAIL'} | "
            f"{item['overall_score']:.3f} | {', '.join(item['tags'])} |"
        )

    failed = [item for item in report["cases"] if not item["passed"]]
    lines += ["", "## Bad Cases", ""]
    if not failed:
        lines.append("No bad cases in this run.")
    else:
        for item in failed:
            lines.append(f"### {item['case_id']}: {item['query']}")
            for metric in item["metrics"]:
                if not metric["passed"]:
                    lines.append(f"- **{metric['name']}**: {metric['reason']}")
            lines.append("")

    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    report = run_suite()
    write_reports(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
