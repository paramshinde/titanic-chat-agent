from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main import app  # noqa: E402


@dataclass(frozen=True)
class VariantCase:
    question: str
    expected_status: int
    check: str


VARIANT_CASES = [
    VariantCase("What percent of travelers were men?", 200, "male_percentage"),
    VariantCase("Give me the male vs female split in percentage.", 200, "male_percentage"),
    VariantCase("Please visualize gender composition with a pie graph.", 200, "has_chart"),
    VariantCase("Could you chart the age distribution?", 200, "has_chart"),
    VariantCase("Draw an age histogram for passengers.", 200, "has_chart"),
    VariantCase("Visualize fare distribution.", 200, "has_chart"),
    VariantCase("Mean fare paid by passengers?", 200, "average_fare"),
    VariantCase("What's the average age aboard?", 200, "average_age"),
    VariantCase("Break down embarkation totals by port.", 200, "embark_counts"),
    VariantCase("Plot embarkation counts as a bar chart.", 200, "has_chart"),
    VariantCase("Overall percent of survivors?", 200, "survival_rate"),
    VariantCase("Display survival outcomes in a chart.", 200, "has_chart"),
    VariantCase("How does survival rate differ across classes?", 200, "survival_by_class"),
    VariantCase("Graph survival rate by class.", 200, "has_chart"),
    VariantCase("Total passenger count?", 200, "total_passengers"),
    VariantCase("Show passenger counts per class as a chart.", 200, "has_chart"),
    VariantCase("import os", 400, "blocked"),
    VariantCase(" ", 400, "empty_invalid"),
]


def _passes_check(check: str, body: dict) -> tuple[bool, str]:
    answer = (body.get("answer") or "").lower()
    chart = body.get("chart")

    if check == "male_percentage":
        return ("gender distribution:" in answer and "male:" in answer), "expected gender percentage answer"
    if check == "has_chart":
        return (isinstance(chart, str) and len(chart) > 100), "expected non-empty chart"
    if check == "average_fare":
        return ("average ticket fare" in answer), "expected average fare answer"
    if check == "average_age":
        return ("average passenger age" in answer), "expected average age answer"
    if check == "embark_counts":
        return ("passengers by embarkation port" in answer), "expected embarkation counts answer"
    if check == "survival_rate":
        return ("overall survival rate" in answer), "expected survival rate answer"
    if check == "survival_by_class":
        return ("survival rate by class" in answer), "expected class survival answer"
    if check == "total_passengers":
        return ("total number of passengers" in answer), "expected total passenger answer"
    if check == "blocked":
        detail = str(body.get("detail", "")).lower()
        return ("blocked content" in detail), "expected blocked content error"
    if check == "empty_invalid":
        return True, "validation handled by FastAPI"
    return False, f"unknown check: {check}"


def main() -> None:
    client = TestClient(app)
    passed = 0
    total = len(VARIANT_CASES)

    print("Running Titanic phrasing-variant suite...\n")
    for idx, case in enumerate(VARIANT_CASES, start=1):
        response = client.post("/ask", json={"question": case.question})
        status_ok = response.status_code == case.expected_status
        body = response.json()
        check_ok, reason = _passes_check(case.check, body)
        ok = status_ok and check_ok
        passed += int(ok)

        status_label = "PASS" if ok else "FAIL"
        print(f"{idx:02d}. [{status_label}] status={response.status_code} expected={case.expected_status}")
        print(f"    Q: {case.question or '[empty question]'}")
        if not ok:
            print(f"    Reason: {reason}")
            print(f"    Response: {json.dumps(body, ensure_ascii=True)}")

    print(f"\nSummary: {passed}/{total} passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
