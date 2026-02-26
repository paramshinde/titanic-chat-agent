import base64
import os
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)

st.set_page_config(page_title="Titanic Chatbot", page_icon="\U0001F6A2", layout="wide")

API_BASE_URL = str(
    st.secrets.get("BACKEND_URL", os.getenv("BACKEND_URL", "http://localhost:8000"))
).strip()

EXAMPLE_QUESTIONS = [
    "What percentage of passengers were male?",
    "Show a histogram of passenger ages",
    "What was the average fare?",
    "How many passengers embarked from each port?",
    "What is the survival rate by class?",
]

UNSUPPORTED_PREFIX = "I can currently answer Titanic dataset questions about:"


@st.cache_data(show_spinner=False)
def fetch_summary() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_BASE_URL}/summary", timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


@st.cache_data(show_spinner=False)
def fetch_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        payload = response.json()
        return payload.get("status") == "ok"
    except requests.RequestException:
        return False


def ask_backend(question: str) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = requests.post(f"{API_BASE_URL}/ask", json={"question": question}, timeout=60)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json(), elapsed_ms


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    if "clear_question_input" not in st.session_state:
        st.session_state["clear_question_input"] = False


def render_sidebar(summary: dict[str, Any] | None, backend_ok: bool) -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-eyebrow">Navigator</div>
            <div class="sidebar-sub">Titanic Analytics Console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"Backend: `{API_BASE_URL}`")

    status_text = "Online" if backend_ok else "Offline"
    status_class = "ok" if backend_ok else "down"
    st.sidebar.markdown(
        f"""
        <div class="sidebar-status {status_class}">
            Backend status: <strong>{status_text}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("#### Suggested Questions")
    for i, sample in enumerate(EXAMPLE_QUESTIONS):
        if st.sidebar.button(sample, key=f"sample_{i}", use_container_width=True):
            st.session_state["question_input"] = sample

    st.sidebar.markdown("#### Dataset Summary")
    if summary is None:
        st.sidebar.warning("Unable to load summary.")
        return

    st.sidebar.markdown(
        f"""
        <div class="sidebar-kpi-grid">
            <div class="sidebar-kpi">
                <div class="sidebar-kpi-label">Rows</div>
                <div class="sidebar-kpi-value">{summary['rows']}</div>
            </div>
            <div class="sidebar-kpi">
                <div class="sidebar-kpi-label">Columns</div>
                <div class="sidebar-kpi-value">{summary['columns']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("Columns"):
        st.write(summary["column_names"])

    with st.sidebar.expander("Missing Values"):
        st.json(summary["missing_values"])


def render_header(summary: dict[str, Any] | None) -> None:
    left, right = st.columns([4, 1])
    with left:
        st.title("\U0001F6A2 Titanic Dataset Chat Agent")
        st.caption("Ask natural language questions and get deterministic answers with optional visual insights.")
    with right:
        st.write("")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    if summary is None:
        st.info("Dataset summary is currently unavailable. Start backend and refresh.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Passengers", summary["rows"])
    with col2:
        st.metric("Columns", summary["columns"])
    with col3:
        missing_total = sum(int(v) for v in summary["missing_values"].values())
        st.metric("Missing Values", missing_total)


def render_quick_actions() -> None:
    st.markdown("#### Quick Prompts")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, sample in enumerate(EXAMPLE_QUESTIONS):
        if cols[i].button(sample, key=f"quick_{i}", use_container_width=True):
            st.session_state["question_input"] = sample


def render_chat() -> None:
    for idx, item in enumerate(st.session_state["chat_history"]):
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            answer = item["answer"]
            st.write(answer)
            meta = f"Response time: {item.get('elapsed_ms', 0):.0f} ms"
            if item.get("chart_bytes"):
                meta += " | Chart generated"
            st.caption(meta)
            if answer.startswith(UNSUPPORTED_PREFIX):
                st.info("That prompt is outside supported analytics types. Try one of the quick prompts.")
            if item.get("chart_bytes"):
                st.image(item["chart_bytes"], caption="Generated chart", use_container_width=True)
                st.download_button(
                    label="Download chart",
                    data=item["chart_bytes"],
                    file_name=f"titanic_chart_{idx}.png",
                    mime="image/png",
                    key=f"download_{idx}",
                )


st.markdown(
    """
    <style>
    :root {
        --bg-main-start: #f8fbff;
        --bg-main-end: #eef4ff;
        --text-primary: #0f172a;
        --primary: #0f4c81;
        --primary-2: #1b5fa7;
        --border-soft: #cfdaea;
        --sidebar-bg: #101a2f;
        --sidebar-text: #e7efff;
        --accent: #0ea5e9;
        --surface: #ffffff;
    }
    .stApp {
        background: radial-gradient(circle at top left, var(--bg-main-start) 0%, var(--bg-main-end) 100%);
        color: var(--text-primary);
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label {
        color: var(--text-primary);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101827 0%, var(--sidebar-bg) 100%);
        border-right: 1px solid #253352;
    }
    div[data-testid="stSidebar"] * {
        color: var(--sidebar-text) !important;
    }
    div[data-testid="stSidebar"] a {
        color: #93c5fd !important;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a;
    }
    div[data-testid="stChatMessage"] {
        background-color: var(--surface);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 0.5rem 0.8rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }
    div[data-baseweb="input"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #b8c7d9 !important;
        border-radius: 10px !important;
    }
    input {
        color: #0f172a !important;
    }
    .stButton > button[kind="primary"],
    button[kind="primary"],
    .stForm button {
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-2) 100%) !important;
        border: 1px solid var(--primary-2) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="secondary"],
    button[kind="secondary"] {
        background: #f8fbff !important;
        border: 1px solid #b8cbe5 !important;
        color: #0f365f !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
    }
    .stSidebar .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid #36507d !important;
        color: #e8f0ff !important;
        text-align: left !important;
        padding: 0.55rem 0.7rem !important;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.12) !important;
    }
    .stSidebar .stAlert {
        background: rgba(14, 165, 233, 0.15) !important;
        border: 1px solid rgba(14, 165, 233, 0.4) !important;
    }
    .sidebar-brand {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid #30405f;
        border-radius: 12px;
        padding: 0.75rem 0.8rem;
        margin-bottom: 0.35rem;
    }
    .sidebar-eyebrow {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f3f7ff;
    }
    .sidebar-sub {
        font-size: 0.8rem;
        color: #a9bddf;
        margin-top: 0.1rem;
    }
    .sidebar-status {
        border-radius: 10px;
        padding: 0.55rem 0.65rem;
        margin: 0.35rem 0 0.8rem 0;
        border: 1px solid transparent;
        font-size: 0.88rem;
    }
    .sidebar-status.ok {
        background: rgba(16, 185, 129, 0.14);
        border-color: rgba(16, 185, 129, 0.5);
    }
    .sidebar-status.down {
        background: rgba(239, 68, 68, 0.16);
        border-color: rgba(239, 68, 68, 0.55);
    }
    .sidebar-kpi-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.45rem;
        margin-bottom: 0.45rem;
    }
    .sidebar-kpi {
        border: 1px solid #30405f;
        border-radius: 10px;
        padding: 0.45rem 0.55rem;
        background: rgba(255, 255, 255, 0.03);
    }
    .sidebar-kpi-label {
        font-size: 0.72rem;
        color: #a8bcde;
    }
    .sidebar-kpi-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #eef4ff;
    }
    .stTextInput input::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_state()
if st.session_state["clear_question_input"]:
    st.session_state["question_input"] = ""
    st.session_state["clear_question_input"] = False

summary_data = fetch_summary()
backend_online = fetch_health()

render_sidebar(summary_data, backend_online)
render_header(summary_data)
render_quick_actions()
render_chat()

with st.form("ask_form"):
    question = st.text_input(
        "Your question",
        key="question_input",
        placeholder="Ask something about Titanic passengers...",
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing dataset..."):
            try:
                payload, elapsed_ms = ask_backend(question.strip())
                chart_bytes = None
                if payload.get("chart"):
                    chart_bytes = base64.b64decode(payload["chart"])

                st.session_state["chat_history"].append(
                    {
                        "question": question.strip(),
                        "answer": payload.get("answer", "No answer returned."),
                        "chart_bytes": chart_bytes,
                        "elapsed_ms": elapsed_ms,
                    }
                )
                st.session_state["clear_question_input"] = True
                st.rerun()
            except requests.HTTPError as exc:
                try:
                    detail = exc.response.json().get("detail", "Request failed.")
                except Exception:
                    detail = "Request failed."
                st.error(detail)
            except requests.RequestException:
                st.error("Could not reach backend. Please check API URL and server status.")
