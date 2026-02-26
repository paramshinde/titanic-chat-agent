from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summary_contract() -> None:
    response = client.get("/summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"] >= 500
    assert payload["columns"] == 12
    assert "column_names" in payload
    assert "missing_values" in payload
    assert "numeric_summary" in payload


def test_ask_example_questions() -> None:
    prompts = [
        "What percentage of passengers were male?",
        "Show a histogram of passenger ages",
        "What was the average fare?",
        "How many passengers embarked from each port?",
    ]
    for prompt in prompts:
        response = client.post("/ask", json={"question": prompt})
        assert response.status_code == 200
        payload = response.json()
        assert payload["answer"].strip()
        if "histogram" in prompt.lower():
            assert payload["chart"] is not None


def test_empty_question_returns_422() -> None:
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_blocked_question_returns_400() -> None:
    response = client.post("/ask", json={"question": "exec(\"print(1)\")"})
    assert response.status_code == 400


def test_unsupported_question_returns_helpful_answer() -> None:
    response = client.post("/ask", json={"question": "Can you summarize the movie plot?"})
    assert response.status_code == 200
    assert response.json()["answer"].startswith("I can currently answer Titanic dataset questions about:")