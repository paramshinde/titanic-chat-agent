from pathlib import Path

import pytest

from agent import TitanicChatAgent
from data_loader import load_titanic_dataframe


@pytest.fixture(scope="module")
def agent() -> TitanicChatAgent:
    df = load_titanic_dataframe(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
    return TitanicChatAgent(df=df)


def test_parse_intent_gender_percentage(agent: TitanicChatAgent) -> None:
    intent = agent.parse_intent("What percentage of passengers were male on the Titanic?")
    assert intent.name == "gender_percentage"


def test_parse_intent_embark_counts_regression(agent: TitanicChatAgent) -> None:
    intent = agent.parse_intent("How many passengers embarked from each port?")
    assert intent.name == "embark_counts"


def test_average_fare_answer(agent: TitanicChatAgent) -> None:
    answer, chart = agent.ask("What was the average ticket fare?")
    assert "average ticket fare" in answer.lower()
    assert chart is None


def test_embark_counts_sum_matches_rows(agent: TitanicChatAgent) -> None:
    answer, _ = agent.ask("How many passengers embarked from each port?")
    assert "Passengers by embarkation port" in answer
    parts = answer.split("->", 1)[1].strip().rstrip(".").split(",")
    total = 0
    for part in parts:
        total += int(part.split(":", 1)[1].strip())
    assert total == len(agent.df)


def test_age_histogram_returns_base64(agent: TitanicChatAgent) -> None:
    answer, chart = agent.ask("Show me a histogram of passenger ages")
    assert "histogram" in answer.lower()
    assert isinstance(chart, str)
    assert len(chart) > 100


def test_non_chart_prompt_has_no_chart(agent: TitanicChatAgent) -> None:
    _, chart = agent.ask("What was the average ticket fare?")
    assert chart is None


def test_unsupported_prompt_returns_help(agent: TitanicChatAgent) -> None:
    answer, chart = agent.ask("Tell me a joke about lifeboats")
    assert answer.startswith("I can currently answer Titanic dataset questions about:")
    assert chart is None


def test_blocked_question_raises_value_error(agent: TitanicChatAgent) -> None:
    with pytest.raises(ValueError):
        agent.ask("import os")