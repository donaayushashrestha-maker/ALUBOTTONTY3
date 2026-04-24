from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your website URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# In-memory store: { session_id: [messages] }
sessions: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # If None, a new session is created


@app.get("/")
def root():
    return {"status": "running"}


# ✅ GET /chat?message=hi&session_id=abc123
@app.get("/chat")
def chat_get(
    message: str = Query(..., description="Your message to the AI"),
    session_id: str | None = Query(None, description="Session ID (auto-generated if not provided)")
):
    return _handle_chat(message, session_id)


# ✅ POST /chat (for frontend fetch/axios calls)
@app.post("/chat")
def chat_post(req: ChatRequest):
    return _handle_chat(req.message, req.session_id)


def _handle_chat(message: str, session_id: str | None):
    try:
        # Create new session if not provided
        session_id = session_id or str(uuid.uuid4())

        # Get or create history for this session
        history = sessions.get(session_id, [])

        # Append new user message
        history.append({"role": "user", "content": message})

        # Build messages with system prompt
        messages_with_system = [
            {"role": "system", "content": "You are a helpful assistant. Remember the conversation and refer back to it when relevant."},
            *history
        ]

        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Free & powerful model on Groq
            messages=messages_with_system,
            max_tokens=1024,
        )

        assistant_reply = response.choices[0].message.content

        # Save assistant reply to history
        history.append({"role": "assistant", "content": assistant_reply})

        # Store updated history
        sessions[session_id] = history

        return {
            "reply": assistant_reply,
            "session_id": session_id,  # Save this to continue the conversation
            "message_count": len(history),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/{session_id}")
def clear_session(session_id: str):
    """Clear memory for a specific session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "cleared", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/chat/{session_id}/history")
def get_history(session_id: str):
    """Get full chat history for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": sessions[session_id]}
