from image_evaluation.runner import run_suite


def test_mvp_suite_all_cases_pass():
    report = run_suite()
    assert report["summary"]["total_cases"] >= 5
    assert report["summary"]["pass_rate"] == 1.0


def test_scores_are_bounded():
    report = run_suite()
    for case in report["cases"]:
        assert 0.0 <= case["overall_score"] <= 1.0
