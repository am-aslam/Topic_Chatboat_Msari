from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
from langdetect import detect
import uuid
import os
import re

MODEL_PATH = r"models\mistral-7b-instruct-v0.1.Q4_0.gguf"

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

# ---------- Relevance guard (hard rule) ----------
_STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","so","to","of","in","on","for",
    "with","at","by","from","as","is","are","was","were","be","been","being","this",
    "that","these","those","it","its","i","you","he","she","we","they","them","my",
    "your","our","their","me","us","do","does","did","doing","have","has","had",
    "will","would","can","could","should","may","might","not","no","yes","too","very"
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def _normalize(text: str) -> set[str]:
    # simple tokenization + lowercase + naive stemming + stopword removal
    tokens = []
    for t in _TOKEN_RE.findall((text or "").lower()):
        # naive stem-ish trimming
        for suf in ("ing","ed","ly","es","s"):
            if len(t) > 4 and t.endswith(suf):
                t = t[: -len(suf)]
                break
        if t and t not in _STOPWORDS and not t.isdigit():
            tokens.append(t)
    return set(tokens)

def is_related(topic: str, message: str, threshold: float = 0.12) -> bool:
    """Return True if message is likely related to topic using Jaccard similarity."""
    topic_set = _normalize(topic)
    msg_set = _normalize(message)
    if not topic_set or not msg_set:
        return True  # don't block empty/degenerate cases
    inter = len(topic_set & msg_set)
    union = len(topic_set | msg_set)
    jaccard = inter / union if union else 0.0
    # Optional: boost if any exact key phrase from topic appears in message
    return jaccard >= threshold

# ---------- Helpers ----------
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

# ---------- Routes ----------
@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session = sessions.setdefault(session_id, {"topic": None, "history": []})

    # Set/replace topic explicitly
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
    user_msg = (req.message or "").strip()

    # HARD GUARD: refuse unrelated queries before calling the model
    if user_msg and not is_related(topic, user_msg):
        reply = "I can only answer questions related to the given topic."
        session["history"].append({"user": user_msg, "bot": reply})
        return {
            "session_id": session_id,
            "topic": topic,
            "reply": reply
        }

    # Language detect (best-effort)
    try:
        lang = detect(user_msg) if user_msg else "en"
    except:
        lang = "en"

    # Short history for grounding
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
- If the user’s message is unrelated, say EXACTLY:
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
        reply = result["choices"][0]["text"].strip() or "I can only answer questions related to the given topic."
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
