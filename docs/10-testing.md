# Chapter 10: Testing an LLM app

**Goal:** automated tests that run in seconds, with no model and no API key, and catch the bugs that matter.

## 10.1 What to test

LLM output changes from run to run, so you don't unit-test *the model*. You test **the code around it**,
which is where this project's important decisions live:

| Test file | Checks |
|---|---|
| `test_tools.py` | ID normalisation, order lookup, the *Processing-only* address rule, ticket numbering, seeding the working copy |
| `test_guardrails.py` | Missed/invented order IDs, false escalations, legal-threat escalation, follow-up IDs from history |
| `test_react_agent.py` | Parser reads the newest step from echoed output; repeated calls blocked; guessed IDs blocked; fallback answer |
| `test_routing.py` | Each message type reaches the right branch, including mislabelled general questions |
| `test_knowledge_base.py` | The FAQ splits into 9 titled sections |

Every test here corresponds to a real bug found while building the project (see
[troubleshooting](troubleshooting.md)). That's the best source of test cases: **when you fix a bug, add a
test so it can't come back.**

## 10.2 Two techniques

**Fake the LLM.** The agent loop calls `react_chain.invoke(...)`. In a test, replace it with an object that
returns pre-written text:

```python
class ScriptedLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
    def invoke(self, inputs):
        return self.replies.pop(0)

monkeypatch.setattr(agent, "react_chain", ScriptedLLM(
    'Thought: look it up\nAction: check_order_status\nAction Input: {"order_id": "ORD1004"}',
    "Thought: I now know the final answer\nFinal Answer: ORD1004 is delayed.",
))
```

Now you can reproduce exactly the misbehaviour you saw (echoing, repeating, guessing IDs) and prove the
loop handles it.

**Isolate the data.** The `temp_data` fixture in `tests/conftest.py` points the tools at a temporary copy of
the orders file, so tests never change your real data.

## 10.3 Run the tests

```powershell
pip install pytest
pytest -v
```

## Checkpoint

```
======================== 24 passed in 2.1s ========================
```

The same tests run automatically on GitHub after every push (chapter 11), using the lightweight
`requirements-dev.txt` so the CI doesn't need to install PyTorch or Whisper.

**Next: [Chapter 11: Publish on GitHub →](11-github.md)**
