from __future__ import annotations

import base64
import io
import re
from typing import Any

import matplotlib.figure
import pandas as pd


BLOCKED_PATTERNS = [
    r"__\w+__",
    r"\bimport\b",
    r"\bopen\(",
    r"\beval\(",
    r"\bexec\(",
    r"\bsubprocess\b",
    r"\bos\.",
    r"\bsys\.",
    r"\brequests\b",
    r"http[s]?://",
    r"`",
]


def sanitize_user_input(question: str) -> str:
    clean = (question or "").strip()
    if not clean:
        raise ValueError("Question cannot be empty.")
    if len(clean) > 500:
        raise ValueError("Question is too long. Keep it under 500 characters.")

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, clean, flags=re.IGNORECASE):
            raise ValueError("Question includes blocked content. Please ask a data question only.")

    return clean


def figure_to_base64(fig: matplotlib.figure.Figure) -> str:
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def build_data_summary(df: pd.DataFrame) -> dict[str, Any]:
    missing = {col: int(df[col].isna().sum()) for col in df.columns}
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    numeric_stats = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().to_dict()
        for col, stats in desc.items():
            numeric_stats[col] = {k: _safe_value(v) for k, v in stats.items()}

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "missing_values": missing,
        "numeric_summary": numeric_stats,
    }