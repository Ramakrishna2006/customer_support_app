"""
The complete Customer Support App: a styled Gradio web UI on top of the full pipeline.

    python app.py      -> open http://127.0.0.1:7860 in your browser

Left: the customer chat (text or voice).
Right: a live "behind the scenes" panel showing how the AI handled each message:
route taken, response time, query analysis, tool calls and FAQ sources.
"""
import base64
import html
import inspect
import os
from pathlib import Path
import threading
import time

import gradio as gr

from config import COMPANY_NAME
from step4_dag_workflow import answer_query
from step5_speech_to_text import transcribe

MODEL_NAME = os.getenv("LLM_MODEL", "openai:gpt-4o-mini").split(":", 1)[-1]

# The ShopEasy logo (assets/logo.svg) is shown in the header and used as the browser-tab icon
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.svg"
LOGO_URI = (
    "data:image/svg+xml;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode()
    if LOGO_PATH.exists() else ""
)

ROUTES = {
    "faq_rag": ("📚", "Knowledge base", "Answered from the help-centre FAQ (RAG)", "#34d399"),
    "quick_lookup": ("⚡", "Quick order lookup", "Order ID found, status fetched directly", "#d4b483"),
    "react_agent": ("🤖", "ReAct agent", "Reasoned step by step and used tools", "#a78bfa"),
    "escalated": ("🚨", "Escalated to a human", "Ticket created for a specialist", "#f87171"),
    "error": ("⚠️", "Error", "Something went wrong", "#8f8a84"),
}
SENTIMENT_EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😟", "angry": "😡"}
URGENCY_COLOR = {"low": "#34d399", "medium": "#d4b483", "high": "#f87171"}

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
/* ===== ShopEasy palette: matches the gold-on-black logo ===== */
:root, body, .dark, .gradio-container, .gradio-container.dark {
  --se-ink: #0a0a0a; --se-ink2: #121212; --se-ink3: #1a1a1a;
  --se-gold: #d4b483; --se-gold2: #a8875a; --se-gold-soft: rgba(212,180,131,.12);
  --se-text: #f3eee8; --se-muted: #8f8a84; --se-line: rgba(255,255,255,.09);

  /* Gradio theme variables, forced dark in both light and dark mode */
  --body-background-fill: var(--se-ink) !important;
  --background-fill-primary: var(--se-ink2) !important;
  --background-fill-secondary: var(--se-ink3) !important;
  --block-background-fill: var(--se-ink2) !important;
  --block-border-color: var(--se-line) !important;
  --border-color-primary: var(--se-line) !important;
  --border-color-accent: rgba(212,180,131,.45) !important;
  --body-text-color: var(--se-text) !important;
  --body-text-color-subdued: var(--se-muted) !important;
  --block-label-text-color: var(--se-muted) !important;
  --block-title-text-color: var(--se-text) !important;
  --block-label-background-fill: var(--se-ink3) !important;
  --input-background-fill: var(--se-ink3) !important;
  --input-background-fill-focus: var(--se-ink3) !important;
  --input-border-color: var(--se-line) !important;
  --input-border-color-focus: var(--se-gold) !important;
  --input-placeholder-color: #6b6660 !important;
  --button-primary-background-fill: var(--se-gold) !important;
  --button-primary-background-fill-hover: #e2c79c !important;
  --button-primary-text-color: var(--se-ink) !important;
  --button-primary-border-color: var(--se-gold) !important;
  --button-secondary-background-fill: transparent !important;
  --button-secondary-background-fill-hover: var(--se-gold-soft) !important;
  --button-secondary-text-color: var(--se-text) !important;
  --button-secondary-border-color: var(--se-line) !important;
  --color-accent: var(--se-gold) !important;
  --color-accent-soft: var(--se-gold-soft) !important;
  --link-text-color: var(--se-gold) !important;
  --table-even-background-fill: var(--se-ink2) !important;
  --table-odd-background-fill: var(--se-ink3) !important;
}
body, gradio-app { background: var(--se-ink) !important; }
.gradio-container { max-width: 1280px !important; margin: auto !important; color: var(--se-text); }
footer { display: none !important; }

/* ----- header ----- */
#hero {
  position: relative; overflow: hidden;
  background: radial-gradient(circle at 88% 0%, rgba(212,180,131,.22), transparent 45%),
              linear-gradient(135deg, #1a1712 0%, #0d0d0d 70%);
  border: 1px solid rgba(212,180,131,.28); border-radius: 22px; padding: 24px 28px; margin-bottom: 6px;
  box-shadow: 0 18px 40px -22px rgba(212,180,131,.45);
}
#hero .brand { display: flex; align-items: center; gap: 18px; }
#hero .logo { width: 68px; height: 68px; border-radius: 18px; flex: none;
  box-shadow: 0 10px 28px -10px rgba(212,180,131,.6); }
#hero .kicker { font-size: 11px; font-weight: 800; letter-spacing: .28em; text-transform: uppercase; color: var(--se-gold); }
#hero h1 { margin: 2px 0 0; font-size: 30px; font-weight: 800; letter-spacing: -.03em; color: var(--se-text); line-height: 1.1; }
#hero h1 span { color: var(--se-gold); }
#hero p { margin: 14px 0 14px; font-size: 15px; color: var(--se-muted); }
#hero .chips span {
  display: inline-block; border: 1px solid rgba(212,180,131,.3); background: rgba(212,180,131,.06);
  padding: 4px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 600; margin: 0 6px 6px 0; color: #e9dcc6;
}
#hero .status { position: absolute; top: 20px; right: 24px; font-size: 12px; font-weight: 600; color: var(--se-text);
  background: rgba(255,255,255,.05); border: 1px solid var(--se-line); padding: 5px 12px; border-radius: 999px; }
#hero .dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#34d399; margin-right:6px;
  box-shadow: 0 0 0 3px rgba(52,211,153,.25); }
@media (max-width: 720px) { #hero .status { position: static; display: inline-block; margin-bottom: 12px; } }

/* ----- chat ----- */
#chatbot { border-radius: 18px !important; border: 1px solid var(--se-line) !important; background: var(--se-ink2) !important; }
#chatbot .message.user, #chatbot [data-testid="user"] {
  background: var(--se-gold) !important; color: var(--se-ink) !important; border-color: var(--se-gold) !important; }
#chatbot .message.bot, #chatbot [data-testid="bot"] {
  background: var(--se-ink3) !important; color: var(--se-text) !important; border-color: var(--se-line) !important; }
#send-btn { min-width: 96px; font-weight: 800 !important; letter-spacing: .04em; }
button.primary, .primary { font-weight: 700 !important; }

/* ----- behind-the-scenes panel ----- */
.panel-title { font-weight: 800; font-size: 12px; margin: 4px 0 10px; letter-spacing: .24em; text-transform: uppercase;
  color: var(--se-gold); display:flex; align-items:center; gap:10px; }
.panel-title:before { content: ""; width: 28px; height: 1px; background: currentColor; }

.card {
  border: 1px solid var(--se-line); background: var(--se-ink2);
  border-radius: 14px; padding: 14px 16px; margin-bottom: 10px; color: var(--se-text);
}
.card .label { font-size: 10.5px; text-transform: uppercase; letter-spacing: .2em; color: var(--se-muted); margin-bottom: 10px; font-weight: 700; }
.empty { color: var(--se-muted); font-size: 13.5px; }

.route { display:flex; align-items:center; gap:12px; }
.route .icon { font-size: 24px; width: 46px; height: 46px; border-radius: 12px; display:flex; align-items:center; justify-content:center; }
.route .name { font-weight: 800; font-size: 16px; }
.route .desc { font-size: 13px; color: var(--se-muted); }
.route .time { margin-left:auto; text-align:right; font-weight:800; font-size: 20px; color: var(--se-gold); }
.route .time small { display:block; font-weight:600; font-size:10.5px; color: var(--se-muted); letter-spacing: .08em; text-transform: uppercase; }

.pills { display:flex; flex-wrap:wrap; gap:8px; }
.pill { border-radius: 10px; padding: 7px 11px; font-size: 13px; border: 1px solid var(--se-line); background: var(--se-ink3); }
.pill b { display:block; font-size: 10px; text-transform: uppercase; letter-spacing:.12em; color: var(--se-muted); font-weight:700; margin-bottom:2px; }
.summary { margin-top:10px; font-size: 13.5px; color: #cfc6ba; font-style: italic; }

.step { border-left: 2px solid var(--se-gold); padding: 4px 0 8px 12px; margin: 0 0 10px 4px; }
.step .head { font-size: 11px; font-weight: 800; letter-spacing: .16em; color: var(--se-gold); margin-bottom: 4px; }
.step .thought { font-size: 13px; font-style: italic; color: #cfc6ba; margin-bottom: 6px; }
.step code { font-size: 12px; background: var(--se-gold-soft); color: #ecd9b8; padding: 2px 6px; border-radius: 6px; }
.step .obs { font-size: 12.5px; margin-top: 6px; padding: 8px 10px; border-radius: 8px; background: var(--se-ink3); color: #d9d2c8; }

.source { display:inline-block; border-radius: 999px; padding: 4px 12px; font-size: 12.5px; margin: 0 6px 6px 0;
  background: var(--se-gold-soft); color: var(--se-gold); border: 1px solid rgba(212,180,131,.3); font-weight: 700; }
.ticket { margin-top: 10px; font-size: 13px; padding: 8px 12px; border-radius: 10px; background: rgba(248,113,113,.1);
  color: #f87171; font-weight: 700; border: 1px solid rgba(248,113,113,.25); }
"""


def make_theme():
    try:
        return gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="stone",
            neutral_hue="stone",
            font=[gr.themes.GoogleFont("Manrope"), "ui-sans-serif", "system-ui", "sans-serif"],
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
        for c in ["LCEL chains", "DAG routing", "ReAct agent", "RAG", "Whisper voice"]
    )
    logo = f'<img class="logo" src="{LOGO_URI}" alt="{esc(COMPANY_NAME)} logo">' if LOGO_URI else ""
    return f"""
<div id="hero">
  <div class="status"><span class="dot"></span>Online · {esc(MODEL_NAME)}</div>
  <div class="brand">
    {logo}
    <div>
      <div class="kicker">AI customer support</div>
      <h1><span>{esc(COMPANY_NAME)}</span> Support Assistant</h1>
    </div>
  </div>
  <p>Ask about orders, refunds, returns, payments or delivery, by typing or speaking.</p>
  <div class="chips">{chips}</div>
</div>"""


def route_html(route: str | None = None, seconds: float | None = None, ticket: str | None = None) -> str:
    if not route:
        return '<div class="card"><div class="label">Route</div><div class="empty">Send a message to see how the AI handles it.</div></div>'
    if route == "working":
        return ('<div class="card"><div class="label">Route</div><div class="route">'
                '<div class="icon" style="background:rgba(212,180,131,.12);border:1px solid rgba(212,180,131,.35)">⏳</div>'
                '<div><div class="name">Working on it…</div><div class="desc">Analysing, searching the FAQ and choosing a route</div></div>'
                '</div></div>')
    icon, name, desc, color = ROUTES.get(route, ROUTES["error"])
    time_block = f'<div class="time">{seconds:.1f}s<small>response time</small></div>' if seconds is not None else ""
    ticket_block = f'<div class="ticket">🎫 {esc(ticket)}</div>' if ticket else ""
    return f"""
<div class="card"><div class="label">Route</div>
  <div class="route">
    <div class="icon" style="background:{color}22;border:1px solid {color}55">{icon}</div>
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
        f"<div style='text-align:center;color:#8f8a84'>"
        + (f"<img src='{LOGO_URI}' width='64' style='border-radius:16px;margin-bottom:10px'><br>" if LOGO_URI else "")
        + f"<b style='color:#f3eee8'>Hi! I'm the {COMPANY_NAME} assistant.</b><br>"
        "Ask me about an order, a refund or our policies.</div>"
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
    if LOGO_PATH.exists() and _accepts(gr.Blocks.launch, "favicon_path"):
        launch_kwargs["favicon_path"] = str(LOGO_PATH)  # logo as the browser-tab icon
    demo.launch(**launch_kwargs)