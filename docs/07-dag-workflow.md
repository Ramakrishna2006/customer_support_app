# Chapter 7: The DAG workflow

**Goal:** combine everything into one pipeline that runs independent steps in parallel and sends each
message down the right branch.

## 7.1 What a DAG is

A **Directed Acyclic Graph** is a workflow whose steps can split apart, run at the same time, and join again,
but never loop back. LangChain builds DAGs from Runnables:

| Runnable | Role in the graph |
|---|---|
| `RunnableParallel(a=..., b=...)` | **Fan-out:** runs several steps on the same input at the same time, returns a dict |
| `RunnableBranch((cond, step), ..., default)` | **Routing:** checks conditions in order, runs the first match |
| `RunnableLambda(fn)` | Turns any Python function into a step |
| `itemgetter("key")` | Passes one field through unchanged |

## 7.2 The design

```
                        query + history
                              │
      ┌───────────────┬───────┴───────┬────────────────┐        STAGE 1: parallel
   query           history        analysis          context
 (pass through) (pass through)   (chapter 3)      (chapter 4 search)
      └───────────────┴───────┬───────┴────────────────┘
                              │
                           ROUTER                                STAGE 2: one branch
     ┌──────────────┬─────────┴─────────┬──────────────────┐
  🚨 ESCALATE    ⚡ QUICK LOOKUP      🤖 REACT AGENT       📚 FAQ / RAG
  needs a human  order ID, just     order ID + action     everything else
  or angry       asking status      (change, return…)
```

Analysis and retrieval don't depend on each other, so they run side by side.

**The routing rules** (checked top to bottom, first match wins):

1. **Escalate** if `needs_human`, or the customer is angry *and* urgency is high.
2. **Quick lookup** if there's an order ID and no action word (*change, cancel, return, refund, broken…*).
3. **ReAct agent** if there's an order ID (so it's an action request).
4. **FAQ** for everything else, including order questions **without** an ID; that branch asks for the ID.

## 7.3 Why the routing looks like this (lessons from real runs)

- **Route on reliable signals.** The first version sent `intent == "order_status"` to the agent. The small
  model labelled *"Do you offer EMI?"* as `order_status`, so the agent ran and guessed an order. Now routing
  uses `order_id` (from the regex guardrail), which is reliable.
- **Don't use an agent when there's only one sensible action.** *"Where is ORD1002?"* always needs exactly one
  tool. Calling it directly takes **1** LLM call instead of 3-6, which cut response time from over 100 s to
  around 20-40 s on a laptop CPU. The agent is kept for requests that need reasoning. This *simple path + agent
  for hard cases* design is common in production.
- **Make RAG answers faithful.** Early FAQ answers said "UPI refunds take 3-5 days" (that's net banking) and
  invented a "Payment Options section". The fixed prompt places the extracts right next to the question, says
  to copy numbers exactly and pick the matching line, forbids mentioning anything not in the extracts, and
  uses `temperature=0`.

## 7.4 The code

Create `step4_dag_workflow.py`:

````python
"""
STEP 4: DAG workflow (parallel branches + conditional routing).

A DAG (Directed Acyclic Graph) is a workflow where steps can split apart,
run at the same time, and join again, but never loop backwards.

                         customer query
                               |
            +------------------+------------------+
            |                  |                  |      STAGE 1: run in PARALLEL
     analyse query       search the FAQ      pass query     (RunnableParallel)
  (intent, sentiment,    (retrieve policy     + history
     urgency...)            context)          through
            |                  |                  |
            +------------------+------------------+
                               |
                           ROUTER                         STAGE 2: pick ONE branch
         +-----------+---------+---------+-----------+       (RunnableBranch)
         |           |                   |           |
     ESCALATE   QUICK LOOKUP        REACT AGENT   FAQ / RAG
   (angry or    (order ID, just     (order ID +    (policy and
   needs human)  asking status)     an action:     general
                 1 tool call,       change, return, questions)
                 1 LLM call         refund...)
         |           |                   |           |
         +-----------+---------+---------+-----------+
                               |
                         final response

Why a QUICK LOOKUP branch? "Where is ORD1002?" doesn't need an agent to *decide*
what to do: there is only one sensible tool. Calling it directly takes 1 LLM call
instead of 3-6, which matters a lot on a laptop CPU. The ReAct agent is kept for
requests that need reasoning or actions. This "simple path + agent for hard cases"
design is common in production systems.

Running analysis and retrieval in parallel cuts waiting time, because neither
depends on the other.
"""
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel

from config import COMPANY_NAME, get_llm
from knowledge_base import search_faq
from step2_structured_output import analysis_chain
from step3_react_agent import run_react_agent

from tools import check_order_status, create_support_ticket


llm = get_llm()

# ---------------------------------------------------------------------------
# STAGE 1: parallel fan-out
# ---------------------------------------------------------------------------
parallel_stage = RunnableParallel(
    query=itemgetter("query"),
    history=itemgetter("history"),
    analysis=analysis_chain,                                       # from Step 2
    context=RunnableLambda(lambda x: search_faq(x["query"])),     # RAG retrieval
)

# ---------------------------------------------------------------------------
# STAGE 2: the four branches
# ---------------------------------------------------------------------------
TONE_GUIDE = {
    "angry": "The customer is upset. Start with a sincere apology and keep it calm and brief.",
    "negative": "The customer is unhappy. Acknowledge the frustration before helping.",
    "neutral": "Be friendly and to the point.",
    "positive": "Match the customer's positive tone.",
}

# Branch 1: FAQ / RAG answer grounded in the retrieved policy text
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are a customer support agent for {COMPANY_NAME}. You answer ONLY from the "
            "help-centre extracts the user gives you.\n"
            "Rules:\n"
            "- Copy numbers, prices, time periods and conditions EXACTLY as written in the extracts. "
            "Pick the line that matches what the customer asked (e.g. UPI vs card vs net banking).\n"
            "- Never mention websites, pages, menus, phone numbers or options that are not in the extracts.\n"
            "- If the extracts don't answer the question, say so and offer to connect them with a human agent.\n"
            "- If they ask about a specific order but haven't given an order ID, ask for it (like ORD1234).\n"
            "- Keep it under 100 words.\n"
            "Tone: {tone}",
        ),
        (
            "human",
            # Small models follow context better when it sits right next to the question
            "Help-centre extracts:\n{context}\n\nConversation so far:\n{history}\n\n"
            "Customer question: {query}\n\nAnswer using only the extracts above.",
        ),
    ]
)
# temperature=0: we want faithful, repeatable answers, not creative ones
rag_answer_chain = rag_prompt | get_llm(temperature=0) | StrOutputParser()


def answer_from_faq(x: dict) -> dict:
    answer = rag_answer_chain.invoke(
        {"query": x["query"], "history": x["history"], "context": x["context"], "tone": TONE_GUIDE[x["analysis"].sentiment]}
    )
    return {"route": "faq_rag", "answer": answer, "agent_steps": [], "ticket": None}


# Branch 2: ReAct agent for anything that needs live data or actions
def answer_with_agent(x: dict) -> dict:
    result = run_react_agent(x["query"], x["history"])
    return {"route": "react_agent", "answer": result["answer"], "agent_steps": result["steps"], "ticket": None}


# Branch 3: quick order lookup (no agent loop needed)
order_reply_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are a customer support agent for {COMPANY_NAME}. Answer the customer's question using "
            "ONLY the order record and help-centre extracts provided. Start with the order status, expected "
            "delivery date and tracking number. Add a policy line only if it is relevant (for example, "
            "what 'Delayed' means). Copy dates and numbers exactly. Under 80 words.\nTone: {tone}",
        ),
        (
            "human",
            "Order record:\n{order_info}\n\nHelp-centre extracts:\n{context}\n\n"
            "Conversation so far:\n{history}\n\nCustomer question: {query}",
        ),
    ]
)
order_reply_chain = order_reply_prompt | get_llm(temperature=0) | StrOutputParser()


def quick_order_lookup(x: dict) -> dict:
    a = x["analysis"]
    order_info = check_order_status.invoke({"order_id": a.order_id})  # plain Python, no LLM needed
    answer = order_reply_chain.invoke(
        {"order_info": order_info, "context": x["context"], "history": x["history"],
         "query": x["query"], "tone": TONE_GUIDE[a.sentiment]}
    )
    step = {
        "thought": "The order ID is known and the customer only wants its status, so the router calls the tool directly.",
        "tool": "check_order_status",
        "tool_input": f'{{"order_id": "{a.order_id}"}}',
        "observation": order_info,
    }
    return {"route": "quick_lookup", "answer": answer, "agent_steps": [step], "ticket": None}


# Branch 4: escalate to a human, with a ticket and an empathetic holding reply
escalation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are a senior support agent at {COMPANY_NAME}. The case below has been escalated to a "
            "human specialist. Write a short, sincere reply (under 80 words): apologise, show you "
            "understand the specific problem, give the ticket ID, and say a specialist will contact "
            "them within 4 hours. Do not promise refunds or outcomes.",
        ),
        ("human", "Issue summary: {summary}\nTicket: {ticket}\nCustomer message: {query}"),
    ]
)
escalation_chain = escalation_prompt | llm | StrOutputParser()


def escalate_to_human(x: dict) -> dict:
    a = x["analysis"]
    ticket = create_support_ticket.invoke({"issue_summary": a.summary, "priority": "high" if a.urgency == "high" else "medium"})
    answer = escalation_chain.invoke({"summary": a.summary, "ticket": ticket, "query": x["query"]})
    return {"route": "escalated", "answer": answer, "agent_steps": [], "ticket": ticket}


# ---------------------------------------------------------------------------
# The router: conditions are checked top to bottom, first match wins
# ---------------------------------------------------------------------------


def should_escalate(x: dict) -> bool:
    a = x["analysis"]
    return a.needs_human or (a.sentiment == "angry" and a.urgency == "high")


# Words that mean the customer wants something DONE, not just information
ACTION_WORDS = (
    "change", "update", "modify", "edit", "cancel", "return", "refund", "replace", "exchange",
    "broken", "damaged", "defective", "wrong", "missing", "address", "ticket", "complain",
)


def should_quick_lookup(x: dict) -> bool:
    """An order ID is known and the customer just wants status information."""
    query = x["query"].lower()
    return bool(x["analysis"].order_id) and not any(word in query for word in ACTION_WORDS)


def should_use_agent(x: dict) -> bool:
    # The agent is only useful when there is a real order to look up. The order ID comes from
    # the regex guardrail in Step 2, which is far more reliable than the LLM's intent label
    # (small models often label general questions as "order_status").
    # Order questions WITHOUT an ID go to the FAQ branch, which asks the customer for the ID.
    return bool(x["analysis"].order_id)


router = RunnableBranch(
    (should_escalate, RunnableLambda(escalate_to_human)),
    (should_quick_lookup, RunnableLambda(quick_order_lookup)),
    (should_use_agent, RunnableLambda(answer_with_agent)),
    RunnableLambda(answer_from_faq),  # default branch
)


def attach_analysis(x: dict) -> dict:
    """Run the router, then add the analysis and retrieved context for display/debugging."""
    result = router.invoke(x)
    result["analysis"] = x["analysis"].model_dump()
    result["context"] = x["context"]
    return result


# The complete DAG
support_pipeline = parallel_stage | RunnableLambda(attach_analysis)


def answer_query(query: str, history: str = "(none)") -> dict:
    """Convenience wrapper used by the app."""
    return support_pipeline.invoke({"query": query, "history": history or "(none)"})


if __name__ == "__main__":
    # LangChain can describe the graph it built. Paste this into https://mermaid.live to see it.
    print(support_pipeline.get_graph().draw_mermaid())

    tests = [
        "How long does a refund take if I paid with UPI?",
        "Where is my order ORD1002?",
        "This is ridiculous! I've been charged twice for ORD1004 and still no product. I want my money back NOW!",
        "Do you offer EMI?",
        "My order ORD1001 arrived broken. Can I return it?",
    ]
    import time

    for q in tests:
        start = time.perf_counter()
        result = answer_query(q)
        seconds = time.perf_counter() - start
        a = result["analysis"]
        print("\n" + "=" * 70)
        print(f"Customer : {q}")
        print(f"Analysis : intent={a['intent']}, sentiment={a['sentiment']}, urgency={a['urgency']}")
        print(f"Order ID : {a['order_id']}   needs_human={a['needs_human']}")
        print(f"Route    : {result['route']}   ({seconds:.1f} s)")
        if result["route"] == "faq_rag":
            # The first line of each retrieved chunk is its FAQ section title
            sections = [chunk.split("\n", 1)[0] for chunk in result["context"].split("\n\n") if chunk.strip()]
            print(f"Sources  : {', '.join(s for s in sections if len(s) < 40)}")
        if result["agent_steps"]:
            tools_used = " -> ".join(f"{s['tool']}({s['tool_input']})" for s in result["agent_steps"])
            print(f"Tools    : {tools_used}")
        if result["ticket"]:
            print(f"Ticket   : {result['ticket']}")
        print(f"Reply    : {result['answer']}")
````

## 7.5 Run it

```powershell
python step4_dag_workflow.py
```

It first prints a **Mermaid** description of the graph. Paste it into [mermaid.live](https://mermaid.live) to
see your pipeline as a diagram (useful for reports). Then it runs 5 test messages.

## Checkpoint

| Message | Route | Correct reply contains |
|---|---|---|
| UPI refund time | `faq_rag` (Sources: Refunds) | 1-3 working days |
| Where is ORD1002? | `quick_lookup` | Shipped, 2026-10-06, TRK88457 |
| Charged twice for ORD1004 | `escalated` | a ticket ID |
| EMI? | `faq_rag` (Sources: Payment methods) | cards above Rs 3,000 |
| ORD1001 arrived broken | `react_agent` | 30 days, return steps |

Each result also prints its **response time**.

**Next: [Chapter 8: Speech-to-text →](08-speech-to-text.md)**
