from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
from langdetect import detect
import uuid
import os


MODEL_PATH = r"models/mistral-7b-openorca.gguf2.Q4_0.gguf"

print(f"🔄 Loading model from {MODEL_PATH} ...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")


llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,      
    n_threads=8,
    chat_format=None
)

print("✅ Model Loaded Successfully!\n")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

sessions = {}

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str | None = None
    set_topic: str | None = None

def trim_history_text(history: str, max_tokens: int = 1000) -> str:
    """Trim conversation history only, not the topic."""
    words = history.split()
    if len(words) > max_tokens:
        history = " ".join(words[-max_tokens:])
    return history

def summarize_topic_if_long(topic: str, max_words: int = 400) -> str:
    """If topic is too long, summarize it inside the prompt."""
    words = topic.split()
    if len(words) <= max_words:
        return topic
    else:

        return " ".join(words[:200]) + " ... " + " ".join(words[-200:])


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session = sessions.setdefault(session_id, {"topic": None, "history": []})


    if req.set_topic:
        session["topic"] = req.set_topic.strip()
        session["history"].clear()
        return {
            "session_id": session_id,
            "topic": session["topic"],
            "reply": f"✅ Topic set successfully! You can now ask questions about: {session['topic']}"
        }


    if not session["topic"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Set topic first: send { set_topic: 'your topic' }"}
        )

    topic = session["topic"]
    user_msg = req.message or ""


    try:
        lang = detect(user_msg)
    except:
        lang = "en"


    history = ""
    for turn in session["history"][-6:]:
        history += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"
    history = trim_history_text(history)


    readable_topic = summarize_topic_if_long(topic)

    prompt = f"""
You are a powerful offline local AI assistant with deep reasoning.
You must answer **only** from the topic below.

TOPIC CONTEXT (read carefully, can be long):
\"\"\"{readable_topic}\"\"\"

RULES:
- Always base answers only on the above topic.
- If the user’s message is unrelated, say:
  "I can only answer questions related to the given topic."
- Respond in detected language: {lang}.
- Give clear, accurate, and detailed answers (not just one-liners).
- You are allowed to reference detailed parts of the topic.
- Never ignore or skip any part of the topic.

Conversation history (shortened):
{history}

User: {user_msg}
Assistant:
""".strip()


    try:
        result = llm(
            prompt,
            max_tokens=512,     
            temperature=0.7,
            stop=["User:", "Assistant:"]
        )
        reply = result["choices"][0]["text"].strip()

    except ValueError:
        reply = "⚠️ The combined topic and conversation are too large for this model. Please simplify or shorten slightly."


    session["history"].append({"user": user_msg, "bot": reply})

    return {
        "session_id": session_id,
        "topic": topic,
        "reply": reply
    }

@app.get("/")
def home():
    return {"status": "✅ Offline Chatbot Backend Running with Full Topic Support!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
