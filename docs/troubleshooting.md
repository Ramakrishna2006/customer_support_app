# Troubleshooting & lessons learned

Every problem below actually happened while building this project on a Windows laptop with Ollama. They're
grouped from "setup" to "AI behaviour".

## Setup and environment

### `source : The term 'source' is not recognized`
`source venv/bin/activate` is for Mac/Linux. On Windows PowerShell use `venv\Scripts\activate`.

### `running scripts is disabled on this system`
Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.

### `OpenAIError: Missing credentials`
The app is trying to use OpenAI. Either:
- `.env` is missing. Check with `dir .env*`; each project folder needs its own `.env`, and `.env` is never in
  a downloaded zip or a Git clone.
- `.env` was saved as `.env.txt` by Notepad: `ren .env.txt .env`.
- `.env` still says `LLM_MODEL=openai:...` but you meant to use Ollama.

Quick way to create a correct Ollama `.env` in PowerShell:

```powershell
@"
LLM_MODEL=ollama:llama3.2
EMBEDDING_MODEL=ollama:nomic-embed-text
COMPANY_NAME=ShopEasy
STT_BACKEND=local
WHISPER_MODEL=base
"@ | Set-Content -Path .env -Encoding ascii
```

### `429 ... insufficient_quota` / `credit_balance_exhausted`
Your OpenAI key works but the account has no credit. Add credit under *Billing*, or switch to Ollama.

### `No module named 'langchain_ollama'`
`pip install langchain-ollama` (inside the activated venv).

### `model "llama3.2" not found` / `model "nomic-embed-text" not found`
`ollama pull llama3.2` and `ollama pull nomic-embed-text`. Check with `ollama list`.

### Connection refused / `httpx.ConnectError`
Ollama isn't running. Open the Ollama app (or run `ollama serve`).

### FFmpeg not found (voice input)
After `winget install ffmpeg`, **close and reopen** the terminal so it sees the new PATH, then activate the
venv again. Or use `STT_BACKEND=openai`.

### My changes don't seem to take effect
You're probably running an old copy of the file. Browsers save repeat downloads as `app (1).py`, and Python
keeps running `app.py`. Check with `dir` and delete the duplicates (see the cleanup command in the
[GitHub chapter](11-github.md) or below).

```powershell
Get-ChildItem -Recurse -File | Where-Object { $_.Name -match ' \(\d+\)' -and $_.FullName -notmatch '\\venv\\' }
```

### A `KeyboardInterrupt` traceback
That's you pressing **Ctrl+C**. On a CPU, a batch or the agent loop can take minutes; let it finish.

## It's slow

On a CPU each LLM call takes roughly 5-25 s depending on the model, so the number of calls per question is
what matters.

| Fix | Effect |
|---|---|
| Use `llama3.2` instead of `llama3.1` | 2-3× faster per call |
| Quick-lookup route for status questions | 1 LLM call instead of 3-6 |
| `keep_alive` in `config.py` | No 10-20 s model reload after 5 idle minutes |
| `num_predict` / `MAX_REPLY_TOKENS` | Shorter replies finish sooner |
| Warm-up thread in `app.py` | First question doesn't pay the loading cost |
| OpenAI `gpt-4o-mini` | 2-5 s per answer |

## AI behaviour: what went wrong and how it was fixed

| What happened | Why | Fix (where) |
|---|---|---|
| Step 1 said *"I've checked your order, it's being processed"* | No data access, so it hallucinated | Tools + agent (ch. 5-6) |
| `order_id: null` for *"Where is ORD1002?"*, and `needs_human: true` | Small model misclassified | Regex & rule guardrails (ch. 3) |
| Agent called the same tool 6 times, then gave up | Model echoed old steps; parser read the first `Action:` | Read the **last** action; block repeats; fallback answer (ch. 6) |
| Unneeded support ticket created | Model over-applied "create a ticket if you can't solve it" | Repeat protection; tickets mainly via the escalation branch |
| EMI question answered with *another customer's* order | Tool docstring contained the example `ORD1002`, which the model copied | Removed example; block IDs the customer never wrote (ch. 5-6) |
| General questions sent to the agent | Routing trusted the fuzzy `intent` label | Route on `order_id` from the guardrail (ch. 7) |
| *"Where is ORD1002?"* answered with address-change policy | Fallback included irrelevant FAQ text | "Answer exactly what was asked" rule; quick-lookup branch (ch. 7) |
| UPI refund "3-5 days" (that's net banking) | Careless paraphrasing of the right passage | Extracts next to the question, "copy numbers exactly", `temperature=0` (ch. 7) |
| Invented a "Payment Options section" | Filled gaps from general knowledge | "Never mention anything not in the extracts" (ch. 7) |
| `<bound method Series.prod ...>` in order text | `row.product` hit pandas' `.prod()` method | Use `row.to_dict()["product"]` (ch. 5) |

**The pattern behind all of these:** small models are good at language and bad at precision. Put the precise
parts (IDs, routing, permissions, repeat limits) in code, give the model only the facts it needs right next to
the question, and test the code paths with a fake LLM.

## Reset everything

```powershell
python reset_data.py
```

Restores the original orders and deletes all tickets.
