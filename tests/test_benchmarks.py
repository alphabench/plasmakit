import json

import pytest

from fusionbench import validate
from fusionbench.benchmarks import CASES, run_case


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_benchmark_case_passes(case):
    result = run_case(case)
    assert result.passed, (
        f"{case.name}: computed {result.computed_value:.4e} vs reference "
        f"{case.reference_value:.4e} ({case.reference}), rel err {result.relative_error:.2e} "
        f"> rtol {case.rtol}"
    )


def test_validate_report(capsys):
    report = validate(verbose=False)
    assert report.passed
    assert len(report.results) == len(CASES)
    assert capsys.readouterr().out == ""


def test_report_str_contains_citations():
    report = validate(verbose=False)
    text = str(report)
    assert "Bosch" in text
    assert "Brysk" in text
    assert "PASS" in text


def test_report_json():
    report = validate(verbose=False)
    record = json.loads(report.to_json())
    assert record["passed"] is True
    assert len(record["results"]) == len(CASES)
