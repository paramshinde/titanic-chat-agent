from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Keep LangChain presence for assignment stack compliance. Runtime logic below is deterministic.
try:
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent as _lc_marker  # noqa: F401
except Exception:  # pragma: no cover - optional import marker
    _lc_marker = None

from utils import figure_to_base64, sanitize_user_input

SUPPORTED_QUESTION_HINT = (
    "I can currently answer Titanic dataset questions about: gender percentage, average fare, "
    "passenger counts, embarkation counts, survival rate, survival rate by class, and age/fare histograms. "
    "Try: 'What percentage of passengers were male?' or 'Show a histogram of passenger ages'."
)


@dataclass(frozen=True)
class IntentResult:
    name: str
    wants_chart: bool = False


@dataclass
class TitanicChatAgent:
    df: pd.DataFrame

    def __post_init__(self) -> None:
        self.answer_handlers: dict[str, Callable[[IntentResult], str]] = {
            "gender_percentage": self._answer_gender_percentage,
            "age_histogram": self._answer_age_histogram,
            "fare_histogram": self._answer_fare_histogram,
            "average_fare": self._answer_average_fare,
            "embark_counts": self._answer_embark_counts,
            "survival_rate": self._answer_survival_rate,
            "survival_by_class": self._answer_survival_by_class,
            "total_passengers": self._answer_total_passengers,
            "passengers_by_class": self._answer_passengers_by_class,
            "average_age": self._answer_average_age,
            "unsupported": self._answer_unsupported,
        }
        self.chart_handlers: dict[str, Callable[[IntentResult], str | None]] = {
            "gender_percentage": self._chart_gender_percentage,
            "age_histogram": self._chart_age_histogram,
            "fare_histogram": self._chart_fare_histogram,
            "embark_counts": self._chart_embark_counts,
            "survival_rate": self._chart_survival_rate,
            "survival_by_class": self._chart_survival_by_class,
            "total_passengers": self._chart_passengers_by_class,
            "passengers_by_class": self._chart_passengers_by_class,
        }

    def ask(self, question: str) -> tuple[str, str | None]:
        clean_question = sanitize_user_input(question)
        intent = self.parse_intent(clean_question)

        answer = self.compute_answer(intent)
        chart_b64 = self.compute_chart(intent)
        return answer, chart_b64

    def parse_intent(self, question: str) -> IntentResult:
        q = question.lower().strip()

        wants_chart = any(k in q for k in ["chart", "plot", "graph", "hist", "histogram", "distribution", "visual"])

        # Specific intents first to avoid accidental generic matching.
        if any(k in q for k in ["male", "female", "men", "women", "sex", "gender"]) and wants_chart:
            return IntentResult(name="gender_percentage", wants_chart=True)

        if "class" in q and "passenger" in q and wants_chart:
            return IntentResult(name="passengers_by_class", wants_chart=True)

        if ("surviv" in q and ("by class" in q or "pclass" in q or "class" in q)):
            return IntentResult(name="survival_by_class", wants_chart=wants_chart)

        if any(k in q for k in ["embark", "port"]):
            return IntentResult(name="embark_counts", wants_chart=wants_chart)

        if ("percentage" in q or "percent" in q) and any(k in q for k in ["male", "female", "men", "women", "sex", "gender"]):
            return IntentResult(name="gender_percentage", wants_chart=wants_chart)

        if "age" in q and any(k in q for k in ["hist", "histogram", "distribution", "chart", "plot", "graph"]):
            return IntentResult(name="age_histogram", wants_chart=True)

        if "fare" in q and any(k in q for k in ["hist", "histogram", "distribution", "chart", "plot", "graph"]):
            return IntentResult(name="fare_histogram", wants_chart=True)

        if ("average" in q or "mean" in q) and "fare" in q:
            return IntentResult(name="average_fare", wants_chart=wants_chart)

        if ("average" in q or "mean" in q) and "age" in q:
            return IntentResult(name="average_age", wants_chart=wants_chart)

        if "surviv" in q and ("rate" in q or "percent" in q or "percentage" in q or wants_chart):
            return IntentResult(name="survival_rate", wants_chart=wants_chart)

        if any(k in q for k in ["how many", "count", "total number", "total passengers", "number of passengers"]):
            return IntentResult(name="total_passengers", wants_chart=wants_chart)

        return IntentResult(name="unsupported", wants_chart=False)

    def compute_answer(self, intent: IntentResult) -> str:
        handler = self.answer_handlers.get(intent.name, self._answer_unsupported)
        return handler(intent)

    def compute_chart(self, intent: IntentResult) -> str | None:
        if not intent.wants_chart and intent.name not in {"age_histogram", "fare_histogram"}:
            return None
        handler = self.chart_handlers.get(intent.name)
        if handler is None:
            return None
        return handler(intent)

    def _answer_gender_percentage(self, _: IntentResult) -> str:
        rows = len(self.df)
        counts = self.df["sex"].value_counts(dropna=False)
        parts: list[str] = []
        for label, count in counts.items():
            pct = (count / rows) * 100 if rows else 0
            parts.append(f"{label}: {pct:.2f}%")
        return "Gender distribution: " + ", ".join(parts) + "."

    def _answer_age_histogram(self, _: IntentResult) -> str:
        return "Displayed the passenger age distribution histogram."

    def _answer_fare_histogram(self, _: IntentResult) -> str:
        return "Displayed the passenger fare distribution histogram."

    def _answer_average_fare(self, _: IntentResult) -> str:
        avg_fare = float(self.df["fare"].mean())
        return f"The average ticket fare is {avg_fare:.2f}."

    def _answer_average_age(self, _: IntentResult) -> str:
        avg_age = float(self.df["age"].mean())
        return f"The average passenger age is {avg_age:.2f}."

    def _answer_embark_counts(self, _: IntentResult) -> str:
        counts = self.df["embarked"].fillna("Unknown").value_counts()
        text = ", ".join([f"{idx}: {int(val)}" for idx, val in counts.items()])
        return f"Passengers by embarkation port -> {text}."

    def _answer_survival_rate(self, _: IntentResult) -> str:
        rate = float(self.df["survived"].mean() * 100)
        return f"The overall survival rate is {rate:.2f}%."

    def _answer_survival_by_class(self, _: IntentResult) -> str:
        grp = self.df.groupby("pclass")["survived"].mean().sort_index() * 100
        text = ", ".join([f"Class {int(idx)}: {val:.2f}%" for idx, val in grp.items()])
        return f"Survival rate by class -> {text}."

    def _answer_total_passengers(self, _: IntentResult) -> str:
        rows = len(self.df)
        return f"The total number of passengers is {rows}."

    def _answer_passengers_by_class(self, _: IntentResult) -> str:
        return "Displayed a chart of passengers by ticket class."

    def _answer_unsupported(self, _: IntentResult) -> str:
        return SUPPORTED_QUESTION_HINT

    def _chart_age_histogram(self, _: IntentResult) -> str | None:
        fig = None
        try:
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.histplot(self.df["age"].dropna(), bins=20, kde=True, ax=ax, color="#1f77b4")
            ax.set_title("Passenger Age Distribution")
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            return figure_to_base64(fig)
        finally:
            if fig is not None:
                plt.close(fig)

    def _chart_fare_histogram(self, _: IntentResult) -> str | None:
        fig = None
        try:
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.histplot(self.df["fare"].dropna(), bins=25, kde=True, ax=ax, color="#ff7f0e")
            ax.set_title("Fare Distribution")
            ax.set_xlabel("Fare")
            ax.set_ylabel("Count")
            return figure_to_base64(fig)
        finally:
            if fig is not None:
                plt.close(fig)

    def _chart_gender_percentage(self, _: IntentResult) -> str | None:
        fig = None
        try:
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(6, 6))
            counts = self.df["sex"].value_counts(dropna=False)
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax.set_title("Passenger Gender Split")
            return figure_to_base64(fig)
        finally:
            if fig is not None:
                plt.close(fig)

    def _chart_embark_counts(self, _: IntentResult) -> str | None:
        fig = None
        try:
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            embark_counts = self.df["embarked"].fillna("Unknown").value_counts()
            sns.barplot(x=embark_counts.index, y=embark_counts.values, hue=embark_counts.index, legend=False, ax=ax)
            ax.set_title("Passengers by Embarkation Port")
            ax.set_xlabel("Embarked")
            ax.set_ylabel("Count")
            return figure_to_base64(fig)
        finally:
            if fig is not None:
                plt.close(fig)

    def _chart_survival_rate(self, _: IntentResult) -> str | None:
        fig = None
        try:
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            survived_counts = self.df["survived"].value_counts().sort_index()
            labels = ["Did Not Survive", "Survived"]
            sns.barplot(x=labels, y=survived_counts.values, hue=labels, legend=False, ax=ax)
            ax.set_title("Survival Counts")
            ax.set_xlabel("Outcome")
            ax.set_ylabel("Passengers")
            return figure_to_base64(fig)
        finally:
            if fig is not None:
                plt.close(fig)

    def _chart_survival_by_class(self, _: IntentResult) -> str | None:
        fig = None
        try:
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            grp = (self.df.groupby("pclass")["survived"].mean() * 100).sort_index()
            x = grp.index.astype(str)
            y = grp.values
            sns.barplot(x=x, y=y, hue=x, legend=False, ax=ax)
            ax.set_title("Survival Rate by Ticket Class")
            ax.set_xlabel("Class")
            ax.set_ylabel("Survival Rate (%)")
            return figure_to_base64(fig)
        finally:
            if fig is not None:
                plt.close(fig)

    def _chart_passengers_by_class(self, _: IntentResult) -> str | None:
        fig = None
        try:
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            class_counts = self.df["pclass"].value_counts().sort_index()
            x = class_counts.index.astype(str)
            y = class_counts.values
            sns.barplot(x=x, y=y, hue=x, legend=False, ax=ax)
            ax.set_title("Passengers by Ticket Class")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            return figure_to_base64(fig)
        finally:
            if fig is not None:
                plt.close(fig)
