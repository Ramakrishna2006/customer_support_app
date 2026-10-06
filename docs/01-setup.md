# Chapter 1: Setup

**Goal:** a project folder with an isolated Python environment, an LLM you can call, and one place
(`config.py`) where every other file gets its models.

## 1.1 Install the tools

| Tool | Why | How |
|---|---|---|
| **Python 3.10+** | The language | [python.org](https://www.python.org/downloads/). On Windows, tick **"Add python.exe to PATH"**. |
| **Git** | Version control, GitHub | [git-scm.com](https://git-scm.com/downloads) |
| **VS Code** (optional) | Editor | [code.visualstudio.com](https://code.visualstudio.com) |
| **Ollama** | Runs LLMs on your computer, free | [ollama.com](https://ollama.com) |
| **FFmpeg** | Audio decoding for Whisper (chapter 8) | `winget install ffmpeg` (Windows) · `brew install ffmpeg` (Mac) · `sudo apt install ffmpeg` (Linux) |

Check them (open a **new** terminal after installing):

```powershell
python --version
git --version
ollama --version
ffmpeg -version
```

## 1.2 Choose your model provider

| | **Ollama (local)** | **OpenAI (cloud)** |
|---|---|---|
| Cost | Free | Pay per use (a few cents for this guide) |
| Speed on a laptop | 10-40 s per answer | 2-5 s per answer |
| Accuracy | Good, makes more mistakes | Very good |
| Privacy | Nothing leaves your computer | Text is sent to OpenAI |
| Setup | Download ~4 GB of models | API key + billing credit |

You can switch any time by editing one line in `.env`. The code doesn't change. This guide assumes Ollama.

**Ollama:** download the chat model and the embedding model (used for search in chapter 4):

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
ollama run llama3.2 "Say hello in one sentence"
```

`llama3.2` (3B parameters) is fast. `llama3.1` (8B) is smarter but 2-3× slower on a CPU.

**OpenAI:** create a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys) and add
credit under *Settings → Billing*. A new account with no credit gives `429 insufficient_quota`.

## 1.3 Create the project and virtual environment

A **virtual environment** (venv) is a private copy of Python for this project, so its libraries don't clash
with other projects.

```powershell
mkdir customer_support_app
cd customer_support_app
python -m venv venv
venv\Scripts\activate
```

Mac/Linux: `source venv/bin/activate`. Your prompt now starts with `(venv)`. **Activate it again every time
you open a new terminal.**

> If PowerShell says *running scripts is disabled*, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

## 1.4 Install the libraries

Create `requirements.txt`:

````text
# ---- LangChain (v1) ----
langchain>=1.0,<2
langchain-core>=1.0,<2
langchain-community>=0.3
langchain-ollama>=0.3        # local models via Ollama (default)
langchain-openai>=0.3        # OpenAI models (optional)

# ---- Speech-to-text ----
openai-whisper               # local Whisper (needs FFmpeg installed on your system)
openai>=1.40                 # OpenAI transcription API (optional)

# ---- Web UI ----
gradio>=5

# ---- Utilities ----
python-dotenv
pydantic>=2
pandas
````

```powershell
pip install -r requirements.txt
```

This takes a while: `openai-whisper` pulls in PyTorch (several hundred MB).

What each library does:

- **langchain / langchain-core:** the framework: prompts, chains, runnables, tools, agents.
- **langchain-ollama / langchain-openai:** connectors that let LangChain talk to each model provider.
- **langchain-community:** extra integrations (not strictly needed, but common in tutorials).
- **openai-whisper:** speech-to-text that runs locally.
- **gradio:** builds a web UI from Python.
- **python-dotenv:** loads settings from `.env`.
- **pydantic:** data models with validation, used for structured LLM output.
- **pandas:** reads and writes the orders CSV.

## 1.5 Settings in `.env`

Settings and secrets (API keys) go in a `.env` file, **never in code**, because code ends up on GitHub.
Create `.env.example` as a template that *is* safe to share:

````ini
# Copy this file to .env and edit it. Never commit .env to GitHub.
#   Windows:     copy .env.example .env
#   Mac/Linux:   cp .env.example .env

# ---- Option A (default): free local models with Ollama ----------------------
# Install Ollama from https://ollama.com, then run:
#   ollama pull llama3.2
#   ollama pull nomic-embed-text
LLM_MODEL=ollama:llama3.2
EMBEDDING_MODEL=ollama:nomic-embed-text

# ---- Option B: OpenAI (paid, faster and more accurate) ----------------------
# Comment out the two Ollama lines above and uncomment these three:
# OPENAI_API_KEY=sk-your-key-here
# LLM_MODEL=openai:gpt-4o-mini
# EMBEDDING_MODEL=openai:text-embedding-3-small

# ---- App settings ------------------------------------------------------------
COMPANY_NAME=ShopEasy
MAX_REPLY_TOKENS=300

# Speech-to-text: "local" (free Whisper on your PC, needs FFmpeg) or "openai" (paid API)
STT_BACKEND=local
WHISPER_MODEL=base
OPENAI_STT_MODEL=whisper-1
````

Then make your real copy:

```powershell
copy .env.example .env
```

Mac/Linux: `cp .env.example .env`.

> ⚠️ Windows Notepad sometimes saves `.env` as `.env.txt`. Check with `dir .env*`. If you see `.env.txt`,
> run `ren .env.txt .env`.

## 1.6 `config.py`: one place to create models

Every step needs an LLM. Instead of creating it in each file, they all call `get_llm()`. Changing provider
then means editing `.env`, nothing else.

````python
"""Shared configuration: loads API keys and creates the models used by every step."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

load_dotenv()  # reads variables from the .env file

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COMPANY_NAME = os.getenv("COMPANY_NAME", "ShopEasy")
MAX_REPLY_TOKENS = int(os.getenv("MAX_REPLY_TOKENS", "300"))


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.3):
    """Return a chat model. Change LLM_MODEL in .env to switch providers.

    Cached, so every step shares the same model objects instead of creating new ones.
    """
    model = os.getenv("LLM_MODEL", "openai:gpt-4o-mini")
    kwargs = {"temperature": temperature}

    if model.startswith("ollama:"):
        # Speed settings for local models:
        kwargs["num_predict"] = MAX_REPLY_TOKENS  # stop long rambling answers early
        kwargs["keep_alive"] = "30m"              # keep the model in memory between messages
        kwargs["num_ctx"] = 4096                  # enough room for prompt + FAQ text
    else:
        kwargs["max_tokens"] = MAX_REPLY_TOKENS

    return init_chat_model(model, **kwargs)


@lru_cache(maxsize=1)
def get_embeddings():
    """Return the embedding model used to search the knowledge base."""
    model = os.getenv("EMBEDDING_MODEL", "openai:text-embedding-3-small")
    return init_embeddings(model)
````

Key points:

- **`init_chat_model("ollama:llama3.2")`** reads the `provider:model` string and creates the right class
  (`ChatOllama`, `ChatOpenAI`, …). That's why one line in `.env` switches providers.
- **`temperature`** controls randomness. Use `0` for classification and facts (same input → same output) and
  ~`0.3` for friendly replies.
- **`@lru_cache`** means calling `get_llm(0)` twice returns the same object, so the program doesn't create
  dozens of model clients.
- **Ollama speed settings:** `keep_alive` stops Ollama unloading the model after 5 idle minutes (reloading
  takes 10-20 s). `num_predict` caps reply length. `num_ctx` gives enough room for the prompt plus FAQ text;
  a too-small context silently cuts off the start of long prompts.

## Checkpoint

```powershell
python -c "from config import get_llm; print(get_llm().invoke('Say hi in 5 words').content)"
```

You should see a short greeting. If you get `Missing credentials`, your `.env` is missing or still says
`openai:`. If you get a connection error, Ollama isn't running: open the Ollama app.

**Next: [Chapter 2: Your first chain →](02-first-chain.md)**
