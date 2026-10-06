"""Shared test setup.

The tests never call a real LLM: they check the plain-Python parts of the app
(tools, guardrails, the ReAct parser and loop, routing rules, FAQ loading).
Model objects are still created when modules are imported, so we point them at
Ollama, which doesn't need an API key or a running server just to be created.
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["LLM_MODEL"] = "ollama:llama3.2"
os.environ["EMBEDDING_MODEL"] = "ollama:nomic-embed-text"

import pytest  # noqa: E402


@pytest.fixture
def temp_data(tmp_path, monkeypatch):
    """Point the tools at a temporary copy of the data, so tests never touch your real files."""
    import tools

    seed = tmp_path / "orders_seed.csv"
    shutil.copy(ROOT / "data" / "orders_seed.csv", seed)
    monkeypatch.setattr(tools, "SEED_PATH", seed)
    monkeypatch.setattr(tools, "ORDERS_PATH", tmp_path / "orders.csv")
    monkeypatch.setattr(tools, "TICKETS_PATH", tmp_path / "tickets.csv")
    return tmp_path
