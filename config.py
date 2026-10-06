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
