from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import TitanicChatAgent
from data_loader import load_titanic_dataframe
from utils import build_data_summary

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_PATH = BASE_DIR.parent / "data" / "titanic.csv"

load_dotenv(ROOT_DIR / ".env", override=True)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class AskResponse(BaseModel):
    answer: str
    chart: str | None = None


app = FastAPI(title="Titanic Dataset Chat Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


df = load_titanic_dataframe(DATA_PATH)
chat_agent = TitanicChatAgent(
    df=df,
)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/summary")
def summary() -> dict:
    return build_data_summary(df)


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    try:
        answer, chart = chat_agent.ask(payload.question)
        return AskResponse(answer=answer, chart=chart)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your question. Please try again.",
        )
