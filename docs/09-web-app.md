# Chapter 9: The web app

**Goal:** a polished browser interface with text and voice chat, plus a live panel that shows *how* the AI
handled each message. That panel is what makes the project easy to demo and explain.

## 9.1 Gradio in two minutes

[Gradio](https://www.gradio.app) builds web UIs from Python. With **`gr.Blocks`** you lay out components in
rows and columns, then connect events (button click, Enter key) to Python functions:

```python
with gr.Blocks() as demo:
    box = gr.Textbox()
    out = gr.Markdown()
    box.submit(my_function, inputs=box, outputs=out)
demo.launch()
```

If the function is a **generator** (uses `yield`), Gradio updates the page each time it yields. The app uses
this to show a *"⏳ Thinking…"* bubble immediately, then replace it with the answer.

## 9.2 What the app does

| Area | Components | Notes |
|---|---|---|
| Header | `gr.HTML` | Gradient banner, feature chips, model status badge |
| Chat | `gr.Chatbot`, `gr.Textbox`, `gr.Button` | Messages are `{"role": ..., "content": ...}` dicts |
| Voice | `gr.Audio(sources=["microphone", "upload"])` | Transcribed with chapter 8, then answered |
| Behind the scenes | `gr.HTML` cards | Route + response time, analysis pills, tool-call timeline, FAQ sources |
| Memory | `format_history` | The last 4 turns are passed to every prompt, so follow-ups work |
| Warm-up | background thread | Loads the model and FAQ index at start-up so the first question is faster |

Styling uses a Gradio **theme** (`gr.themes.Soft`) plus custom CSS. The cards use Gradio's CSS variables
(`--block-background-fill`, `--border-color-primary`), so they look right in light and dark mode.

> **Version note:** Gradio 6 moved `theme=` and `css=` from `gr.Blocks()` to `demo.launch()`, and removed
> `Chatbot(type="messages")`. The app checks which version is installed (`_accepts`) and passes them in the
> right place, so it runs on Gradio 5 and 6.

## 9.3 The code

Create `app.py`:

````python
"""
The complete Customer Support App: a styled Gradio web UI on top of the full pipeline.

    python app.py      -> open http://127.0.0.1:7860 in your browser

Left: the customer chat (text or voice).
Right: a live "behind the scenes" panel showing how the AI handled each message:
route taken, response time, query analysis, tool calls and FAQ sources.
"""
import html
import inspect
import os
import threading
import time

import gradio as gr

from config import COMPANY_NAME
from step4_dag_workflow import answer_query
from step5_speech_to_text import transcribe

MODEL_NAME = os.getenv("LLM_MODEL", "openai:gpt-4o-mini").split(":", 1)[-1]

ROUTES = {
    "faq_rag": ("📚", "Knowledge base", "Answered from the help-centre FAQ (RAG)", "#059669"),
    "quick_lookup": ("⚡", "Quick order lookup", "Order ID found, status fetched directly", "#2563eb"),
    "react_agent": ("🤖", "ReAct agent", "Reasoned step by step and used tools", "#7c3aed"),
    "escalated": ("🚨", "Escalated to a human", "Ticket created for a specialist", "#dc2626"),
    "error": ("⚠️", "Error", "Something went wrong", "#6b7280"),
}
SENTIMENT_EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😟", "angry": "😡"}
URGENCY_COLOR = {"low": "#059669", "medium": "#d97706", "high": "#dc2626"}

EXAMPLES = [
    "Where is my order ORD1002?",
    "How long does a UPI refund take?",
    "Do you offer EMI?",
    "Please change the address on ORD1003 to 21 Park Street, Kolkata",
    "I've been charged twice for ORD1004 and still no product. I want my money back NOW!",
]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
.gradio-container { max-width: 1280px !important; margin: auto !important; }
footer { display: none !important; }

#hero {
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
  border-radius: 20px; padding: 26px 30px; color: #fff; margin-bottom: 6px;
  box-shadow: 0 12px 32px -12px rgba(79, 70, 229, .55);
}
#hero h1 { margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -.02em; color: #fff; }
#hero p { margin: 6px 0 14px; opacity: .92; font-size: 15px; color: #fff; }
#hero .chips span {
  display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3);
  padding: 4px 11px; border-radius: 999px; font-size: 12.5px; margin: 0 6px 6px 0; color: #fff;
}
#hero .status { float: right; font-size: 12.5px; background: rgba(0,0,0,.22); padding: 5px 12px; border-radius: 999px; }
#hero .dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#4ade80; margin-right:6px;
  box-shadow: 0 0 0 3px rgba(74,222,128,.3); }

.panel-title { font-weight: 700; font-size: 15px; margin: 2px 0 8px; display:flex; align-items:center; gap:8px; }

.card {
  border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
  border-radius: 14px; padding: 14px 16px; margin-bottom: 10px;
}
.card .label { font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em; opacity: .65; margin-bottom: 8px; font-weight: 600; }
.empty { opacity: .55; font-size: 13.5px; }

.route { display:flex; align-items:center; gap:12px; }
.route .icon { font-size: 26px; width: 46px; height: 46px; border-radius: 12px; display:flex; align-items:center; justify-content:center; }
.route .name { font-weight: 700; font-size: 16px; }
.route .desc { font-size: 13px; opacity: .75; }
.route .time { margin-left:auto; text-align:right; font-weight:700; font-size: 18px; }
.route .time small { display:block; font-weight:500; font-size:11px; opacity:.6; }

.pills { display:flex; flex-wrap:wrap; gap:8px; }
.pill { border-radius: 10px; padding: 7px 11px; font-size: 13px; border: 1px solid var(--border-color-primary); }
.pill b { display:block; font-size: 10.5px; text-transform: uppercase; letter-spacing:.05em; opacity:.6; font-weight:600; margin-bottom:2px; }
.summary { margin-top:10px; font-size: 13.5px; opacity: .85; font-style: italic; }

.step { border-left: 3px solid #7c3aed; padding: 4px 0 8px 12px; margin: 0 0 10px 4px; }
.step .head { font-size: 12px; font-weight: 700; color: #7c3aed; margin-bottom: 4px; }
.step .thought { font-size: 13px; font-style: italic; opacity: .8; margin-bottom: 6px; }
.step code { font-size: 12px; background: rgba(124,58,237,.1); padding: 2px 6px; border-radius: 6px; }
.step .obs { font-size: 12.5px; margin-top: 6px; padding: 8px 10px; border-radius: 8px; background: rgba(127,127,127,.08); }

.source { display:inline-block; border-radius: 999px; padding: 4px 11px; font-size: 12.5px; margin: 0 6px 6px 0;
  background: rgba(5,150,105,.12); color: #059669; border: 1px solid rgba(5,150,105,.25); font-weight: 600; }
.ticket { margin-top: 10px; font-size: 13px; padding: 8px 12px; border-radius: 10px; background: rgba(220,38,38,.1);
  color: #dc2626; font-weight: 600; }

#chatbot { border-radius: 16px !important; }
#send-btn { min-width: 90px; }
"""


def make_theme():
    try:
        return gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="violet",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML renderers for the "behind the scenes" panel
# ---------------------------------------------------------------------------
esc = html.escape


def hero_html() -> str:
    chips = "".join(
        f"<span>{c}</span>"
        for c in ["🔗 LCEL chains", "🧭 DAG routing", "🤖 ReAct agent", "📚 RAG", "🎤 Whisper voice"]
    )
    return f"""
<div id="hero">
  <div class="status"><span class="dot"></span>Online · {esc(MODEL_NAME)}</div>
  <h1>🛍️ {esc(COMPANY_NAME)} Support Assistant</h1>
  <p>Ask about orders, refunds, returns, payments or delivery, by typing or speaking.</p>
  <div class="chips">{chips}</div>
</div>"""


def route_html(route: str | None = None, seconds: float | None = None, ticket: str | None = None) -> str:
    if not route:
        return '<div class="card"><div class="label">Route</div><div class="empty">Send a message to see how the AI handles it.</div></div>'
    if route == "working":
        return ('<div class="card"><div class="label">Route</div><div class="route">'
                '<div class="icon" style="background:rgba(79,70,229,.12)">⏳</div>'
                '<div><div class="name">Working on it…</div><div class="desc">Analysing, searching the FAQ and choosing a route</div></div>'
                '</div></div>')
    icon, name, desc, color = ROUTES.get(route, ROUTES["error"])
    time_block = f'<div class="time">{seconds:.1f}s<small>response time</small></div>' if seconds is not None else ""
    ticket_block = f'<div class="ticket">🎫 {esc(ticket)}</div>' if ticket else ""
    return f"""
<div class="card"><div class="label">Route</div>
  <div class="route">
    <div class="icon" style="background:{color}1f">{icon}</div>
    <div><div class="name" style="color:{color}">{name}</div><div class="desc">{desc}</div></div>
    {time_block}
  </div>{ticket_block}
</div>"""


def analysis_html(a: dict | None) -> str:
    if not a:
        return '<div class="card"><div class="label">Query analysis</div><div class="empty">Intent, sentiment and urgency will appear here.</div></div>'
    intent = a.get("intent", "").replace("_", " ").title()
    sentiment = a.get("sentiment", "")
    urgency = a.get("urgency", "")
    pills = [
        ("Intent", esc(intent), None),
        ("Sentiment", f"{SENTIMENT_EMOJI.get(sentiment, '')} {esc(sentiment.title())}", None),
        ("Urgency", esc(urgency.title()), URGENCY_COLOR.get(urgency)),
        ("Order ID", esc(a.get("order_id") or "—"), None),
        ("Human needed", "Yes" if a.get("needs_human") else "No", "#dc2626" if a.get("needs_human") else None),
    ]
    pills_html = "".join(
        f'<div class="pill"><b>{label}</b><span style="{f"color:{color};font-weight:700" if color else ""}">{value}</span></div>'
        for label, value, color in pills
    )
    summary = f'<div class="summary">“{esc(a.get("summary", ""))}”</div>' if a.get("summary") else ""
    return f'<div class="card"><div class="label">Query analysis · structured output</div><div class="pills">{pills_html}</div>{summary}</div>'


def steps_html(steps: list | None) -> str:
    if not steps:
        return '<div class="card"><div class="label">Tool calls</div><div class="empty">No tools were needed for this reply.</div></div>'
    items = []
    for i, s in enumerate(steps, 1):
        obs = s.get("observation", "")
        obs = obs if len(obs) < 320 else obs[:320] + "…"
        thought = f'<div class="thought">💭 {esc(s.get("thought") or "")}</div>' if s.get("thought") else ""
        items.append(
            f'<div class="step"><div class="head">STEP {i}</div>{thought}'
            f'🔧 <code>{esc(s.get("tool", ""))}</code> <code>{esc(s.get("tool_input", ""))}</code>'
            f'<div class="obs">👁️ {esc(obs)}</div></div>'
        )
    return f'<div class="card"><div class="label">Tool calls · Thought → Action → Observation</div>{"".join(items)}</div>'


def sources_html(context: str | None, route: str | None) -> str:
    if not context or route not in ("faq_rag", "quick_lookup"):
        return '<div class="card"><div class="label">Knowledge-base sources</div><div class="empty">FAQ sections used for the answer will appear here.</div></div>'
    titles = [chunk.split("\n", 1)[0] for chunk in context.split("\n\n") if chunk.strip()]
    chips = "".join(f'<span class="source">📄 {esc(t)}</span>' for t in titles if len(t) < 40)
    return f'<div class="card"><div class="label">Knowledge-base sources · RAG</div>{chips}</div>'


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------
def _text(content) -> str:
    """Gradio versions store message content differently; always get plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    if isinstance(content, (list, tuple)):
        return " ".join(_text(c) for c in content)
    return str(content)


def format_history(history: list, max_turns: int = 4) -> str:
    """Recent chat turns as plain text for the prompts (conversation memory)."""
    lines = []
    for m in history[-max_turns * 2:]:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "user")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        text = _text(content).replace("🎤 ", "")
        lines.append(f"{'Customer' if role == 'user' else 'Agent'}: {text}")
    return "\n".join(lines) or "(none)"


EMPTY_PANELS = (route_html(), analysis_html(None), steps_html(None), sources_html(None, None), {})
THINKING = "⏳ *Thinking…*"


def respond(query: str, shown_text: str, history: list):
    """Generator: first shows a 'thinking' bubble, then the real answer."""
    history = list(history or [])
    memory = format_history(history)
    history.append({"role": "user", "content": shown_text})
    yield history + [{"role": "assistant", "content": THINKING}], route_html("working"), gr.update(), gr.update(), gr.update(), gr.update()

    start = time.perf_counter()
    try:
        result = answer_query(query, memory)
        seconds = time.perf_counter() - start
        history.append({"role": "assistant", "content": result["answer"]})
        yield (
            history,
            route_html(result["route"], seconds, result.get("ticket")),
            analysis_html(result["analysis"]),
            steps_html(result["agent_steps"]),
            sources_html(result["context"], result["route"]),
            result["analysis"],
        )
    except Exception as exc:
        history.append({"role": "assistant", "content": f"⚠️ Sorry, something went wrong: `{exc}`\n\nIs Ollama running, or is your API key set in `.env`?"})
        yield history, route_html("error"), analysis_html(None), steps_html(None), sources_html(None, None), {}


def on_text(message: str, history: list):
    if not message or not message.strip():
        yield history, "", *[gr.update()] * 5
        return
    for out in respond(message.strip(), message.strip(), history):
        yield out[0], "", *out[1:]


def on_voice(audio_path: str, history: list):
    history = list(history or [])
    if not audio_path:
        gr.Warning("Record or upload an audio clip first.")
        yield history, None, *[gr.update()] * 5
        return
    yield history + [{"role": "assistant", "content": "🎤 *Transcribing your voice message…*"}], None, *[gr.update()] * 5
    try:
        text = transcribe(audio_path)
    except Exception as exc:
        gr.Warning(f"Transcription failed: {exc}")
        yield history, None, *[gr.update()] * 5
        return
    if not text:
        gr.Warning("I couldn't hear any speech in that recording.")
        yield history, None, *[gr.update()] * 5
        return
    for out in respond(text, f"🎤 {text}", history):
        yield out[0], None, *out[1:]


def clear_all():
    return [], "", *EMPTY_PANELS


def warm_up():
    """Load the model and build the FAQ index in the background so the first question is faster."""
    try:
        from config import get_llm
        from knowledge_base import get_vector_store

        get_vector_store()
        get_llm(temperature=0).invoke("Reply with OK.")
        print("✅ Model warmed up and knowledge base ready.")
    except Exception as exc:
        print(f"(warm-up skipped: {exc})")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def _accepts(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


# Gradio 6 moved theme/css from Blocks() to launch(); support both versions
THEME = make_theme()
STYLE_IN_LAUNCH = _accepts(gr.Blocks.launch, "css")
blocks_kwargs = {"title": f"{COMPANY_NAME} Support"}
if not STYLE_IN_LAUNCH:
    blocks_kwargs.update(css=CSS, theme=THEME)

chatbot_kwargs = {"height": 520, "show_label": False, "elem_id": "chatbot"}
if _accepts(gr.Chatbot.__init__, "type"):
    chatbot_kwargs["type"] = "messages"  # Gradio 5
if _accepts(gr.Chatbot.__init__, "placeholder"):
    chatbot_kwargs["placeholder"] = (
        f"<div style='text-align:center;opacity:.7'><div style='font-size:42px'>👋</div>"
        f"<b>Hi! I'm the {COMPANY_NAME} assistant.</b><br>Ask me about an order, a refund or our policies.</div>"
    )

with gr.Blocks(**blocks_kwargs) as demo:
    gr.HTML(hero_html())

    with gr.Row():
        # ----- Chat column -----
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(**chatbot_kwargs)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your question… e.g. Where is my order ORD1002?",
                    show_label=False, scale=6, container=False, autofocus=True,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1, elem_id="send-btn")
            with gr.Accordion("🎤 Ask with your voice", open=False):
                audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", show_label=False)
                voice_btn = gr.Button("Send voice message", variant="secondary")
            gr.Examples(EXAMPLES, inputs=msg, label="💡 Try one of these")
            clear_btn = gr.Button("🗑️ New conversation", variant="secondary", size="sm")

        # ----- Insights column -----
        with gr.Column(scale=5):
            gr.HTML('<div class="panel-title">🔍 Behind the scenes</div>')
            route_box = gr.HTML(route_html())
            analysis_box = gr.HTML(analysis_html(None))
            steps_box = gr.HTML(steps_html(None))
            sources_box = gr.HTML(sources_html(None, None))
            with gr.Accordion("Raw analysis JSON", open=False):
                raw_json = gr.JSON(show_label=False)

    panels = [route_box, analysis_box, steps_box, sources_box, raw_json]
    msg.submit(on_text, [msg, chatbot], [chatbot, msg, *panels])
    send_btn.click(on_text, [msg, chatbot], [chatbot, msg, *panels])
    voice_btn.click(on_voice, [audio_in, chatbot], [chatbot, audio_in, *panels])
    clear_btn.click(clear_all, None, [chatbot, msg, *panels])


if __name__ == "__main__":
    threading.Thread(target=warm_up, daemon=True).start()
    launch_kwargs = {"css": CSS, "theme": THEME} if STYLE_IN_LAUNCH else {}
    demo.launch(**launch_kwargs)
````

## 9.4 Run it

```powershell
python app.py
```

Wait for `✅ Model warmed up and knowledge base ready.`, then open **http://127.0.0.1:7860**.

## Checkpoint

Try these and watch the **Behind the scenes** panel:

1. Click the example questions. Each shows a different route card colour.
2. *"Where is my order ORD1006?"*, then *"When will it arrive?"*: the second has no ID, but the guardrail
   finds ORD1006 in the chat history.
3. Open **🎤 Ask with your voice**, record a question, click **Send voice message**. The first voice message
   downloads the Whisper model (~140 MB).

Press **Ctrl+C** in the terminal to stop the app.

**Next: [Chapter 10: Testing →](10-testing.md)**
