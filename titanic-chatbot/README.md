# Titanic Dataset Chat Agent

A deterministic Titanic analytics chatbot that answers natural-language questions with text insights and optional charts.

## Tech Stack

- Backend: FastAPI + Pandas
- Agent layer: LangChain package included (runtime logic is deterministic and does not require model keys)
- Frontend: Streamlit
- Visualization: Matplotlib + Seaborn

## Features

- Natural language Titanic Q&A (deterministic pandas-first logic)
- Accurate answers for assignment-style prompts
- Chart generation for supported visualization intents
- Clean Streamlit chat UI with chart preview + download
- API endpoints for health and dataset summary
- No API key required to run

## Project Structure

```text
titanic-chatbot/
|
+-- backend/
|   +-- main.py
|   +-- agent.py
|   +-- data_loader.py
|   +-- utils.py
|   +-- requirements.txt
|   +-- tests/
|       +-- test_agent.py
|       +-- test_api.py
|
+-- frontend/
|   +-- app.py
|   +-- requirements.txt
|
+-- data/
|   +-- titanic.csv
|
+-- .env.example
+-- README.md
```

## Setup

### 1) Prepare environment file

```bash
cd titanic-chatbot
cp .env.example .env
```

No model/API keys are needed for core functionality.

### 2) Run backend (FastAPI)

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend: `http://localhost:8000`

### 3) Run frontend (Streamlit)

Open a second terminal:

```bash
cd titanic-chatbot/frontend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Frontend: `http://localhost:8501`

## API Contract

### `POST /ask`

Request:

```json
{
  "question": "Show a histogram of passenger ages"
}
```

Response:

```json
{
  "answer": "Displayed the passenger age distribution histogram.",
  "chart": "<base64_png_or_null>"
}
```

### `GET /summary`

Returns dataset shape, columns, missing counts, and numeric summary.

### `GET /health`

Returns:

```json
{"status": "ok"}
```

## Supported Question Types

- Gender percentage (male/female split)
- Average fare
- Average age
- Total passengers
- Embarkation counts by port
- Survival rate (overall)
- Survival rate by class
- Age histogram
- Fare histogram

Unsupported prompts return a helpful guidance response instead of failing.

## Assignment Example Questions

- What percentage of passengers were male on the Titanic?
- Show me a histogram of passenger ages
- What was the average ticket fare?
- How many passengers embarked from each port?

## Run Tests

```bash
cd titanic-chatbot/backend
pytest -q
```

Test suite covers:

- Intent parsing and regression checks
- Deterministic analytics outputs
- Chart generation
- API endpoint contract and error handling

## Notes

- The included dataset is the full Titanic train-style CSV (891 rows, 12 columns).
- LangChain is included in dependencies for stack alignment, but runtime answers are deterministic and do not depend on external model providers.