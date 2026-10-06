# 📖 Build the Customer Support Assistant from scratch

This guide rebuilds the whole project from an empty folder, one concept at a time. Each chapter explains
**why** before **how**, gives you the code, tells you what output to expect, and lists the problems you're
likely to hit.

## Before you start

- **Level:** you know basic Python (functions, dictionaries, classes). No prior LangChain or AI experience needed.
- **Time:** about 4-6 hours in total, spread over a few sessions.
- **Cost:** free with Ollama. With OpenAI, a few cents for the whole guide.
- **Computer:** 8 GB RAM minimum for local models (16 GB is more comfortable). Any OS.

## The learning path

| # | Chapter | What you build | Key ideas |
|---|---|---|---|
| 1 | [Setup](01-setup.md) | Project folder, venv, `.env`, `config.py` | Virtual environments, API keys, model providers |
| 2 | [Your first chain](02-first-chain.md) | `step1_basic_chain.py` | Prompt templates, LCEL `\|`, invoke/batch/stream |
| 3 | [Structured output](03-structured-output.md) | `step2_structured_output.py` | Pydantic schemas, classification, guardrails |
| 4 | [Knowledge base (RAG)](04-knowledge-base-rag.md) | `data/faq.md`, `knowledge_base.py` | Chunking, embeddings, similarity search |
| 5 | [Tools](05-tools.md) | `data/orders_seed.csv`, `tools.py` | `@tool`, docstrings, safe side effects |
| 6 | [ReAct agent](06-react-agent.md) | `step3_react_agent.py` | Reason + Act loop, parsing, loop protection |
| 7 | [DAG workflow](07-dag-workflow.md) | `step4_dag_workflow.py` | Parallel steps, conditional routing |
| 8 | [Speech-to-text](08-speech-to-text.md) | `step5_speech_to_text.py` | Whisper, plugging non-LLM steps into chains |
| 9 | [Web app](09-web-app.md) | `app.py` | Gradio Blocks, live status, styling |
| 10 | [Testing](10-testing.md) | `tests/` | Testing LLM apps without an LLM |
| 11 | [Publish on GitHub](11-github.md) | `.gitignore`, CI workflow | Git, secrets, GitHub Actions |
| — | [Troubleshooting & lessons learned](troubleshooting.md) | | What actually went wrong, and why |

## The one idea that ties it all together

> **Let the LLM handle language. Let plain code handle the things that must be right.**

The LLM is great at understanding messy human messages and writing friendly replies. It is unreliable at
copying an order ID, deciding when to stop, or never inventing a fact, especially a small local model.
Throughout this project, every important decision (routing, order IDs, which actions are allowed, when to
escalate) is checked or made by ordinary Python. You'll see this pattern in chapters 3, 6 and 7.

## Conventions

- Commands are shown for **Windows PowerShell**; Mac/Linux equivalents are given where they differ.
- `(venv)` at the start of your prompt means the virtual environment is active. Most commands need it.
- Every chapter ends with **Checkpoint**: what you should see before moving on.

Start with **[Chapter 1: Setup →](01-setup.md)**
