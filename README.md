# 🧠 Offline Topic ChatBot (FastAPI + Llama.cpp)

An offline, privacy-focused **AI Topic Chatbot** built using **FastAPI** and **Llama.cpp**.  
This chatbot runs fully offline, detects language automatically, supports topic-based conversations, and assigns unique session IDs to maintain context.

---

## 🚀 Features

- ✅ **Works Completely Offline** – No internet required
- 🧠 **Llama.cpp GGUF model** for fast inference
- 🌍 **Automatic Language Detection** (Multi-language support)
- 🧵 **Session-based conversations** by topic
- ⚡ Lightweight — Works on CPU
- 🔌 Can be integrated with Web UI, Desktop UI, or Mobile
- 🛡️ **Fully Private** – Your data stays on your system

---

## 📂 Project Structure

ChatBoat/
│── index.html # Frontend UI
│── main.py # FastAPI Backend
│── requirements.txt # Python Dependencies
│── models/ # Place your GGUF Model here
└── venv/ # (Optional) Virtual Environment


---

## 📥 Download AI Model (Required)

Download the GGUF model and place it inside the `models` folder.

🔗 **Model Download Link:**  
https://huggingface.co/am-as1am/Llama-3.2-3B-Instruct-Q4_0/blob/main/mistral-7b-openorca.gguf2.Q4_0.gguf

> After downloading, move the model into the `models` directory and make sure the filename matches the path used inside `main.py`.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd ChatBoat

2️⃣ Create & Activate Virtual Environment (Recommended)
python -m venv venv


Activate it:

OS	Command
Windows	venv\Scripts\activate
Mac/Linux	source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Run the Backend Server
python main.py
