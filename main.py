from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import os

app = FastAPI()

# Allow your website to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your website URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # Optional conversation history


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        messages = req.history + [{"role": "user", "content": req.message}]

        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=messages,
        )

        return {
            "reply": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
