"""
STEP 2: Structured output (intent, sentiment, urgency).

Free text is fine for replies, but the app needs to make DECISIONS:
which team handles this? is the customer angry? is it urgent?
For that we ask the LLM to fill in a Pydantic model instead of writing prose.
`.with_structured_output()` guarantees we get a typed Python object back.
"""
import re
from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from config import COMPANY_NAME, get_llm


class QueryAnalysis(BaseModel):
    """Analysis of a customer support message."""

    intent: Literal[
        "order_status",
        "address_change",
        "returns_refunds",
        "billing_payment",
        "technical_issue",
        "account_login",
        "complaint",
        "general_question",
    ] = Field(description="The main thing the customer wants")
    sentiment: Literal["positive", "neutral", "negative", "angry"] = Field(
        description="The customer's emotional tone"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="high = money lost, fraud, safety, or a very upset customer"
    )
    order_id: Optional[str] = Field(
        default=None, description="Order ID mentioned in the message, like ORD1002, otherwise null"
    )
    needs_human: bool = Field(
        description="True if a human must handle it: fraud, legal threats, repeated failures, or explicit request for a human"
    )
    summary: str = Field(description="One-sentence summary of the issue for the support team")


analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You analyse customer messages for {COMPANY_NAME}'s support team. "
            "Read the conversation history for context, then classify the LATEST message.\n\n"
            "Rules:\n"
            "- order_id: copy any ID like ORD1002 exactly as written; null only if none appears.\n"
            "- urgency: low = general questions; medium = a problem with one order; "
            "high = money lost, fraud, safety, or an angry customer.\n"
            "- needs_human: true ONLY for fraud, legal threats, repeated failures, or when the customer "
            "asks for a person. Order status, returns, refunds and address changes are handled by "
            "the bot, so needs_human is false for those.\n\n"
            "Example: 'Where is ORD1006?' -> intent=order_status, sentiment=neutral, urgency=medium, "
            "order_id=ORD1006, needs_human=false",
        ),
        ("human", "Conversation history:\n{history}\n\nLatest message:\n{query}"),
    ]
)

# temperature=0 makes classification consistent
llm_analysis_chain = analysis_prompt | get_llm(temperature=0).with_structured_output(QueryAnalysis)

# ---------------------------------------------------------------------------
# Guardrails: simple rules that catch mistakes the LLM makes.
# Small models often miss obvious facts, so we don't trust them blindly.
# ---------------------------------------------------------------------------
ORDER_ID_RE = re.compile(r"\bORD[\s\-#]*(\d{3,})\b", re.IGNORECASE)
ESCALATION_WORDS = (
    "human", "real person", "agent", "manager", "supervisor", "complaint", "court",
    "legal", "lawyer", "police", "fraud", "scam", "hacked", "unauthori", "third time",
    "again and again", "charged twice", "double charged",
)


def apply_guardrails(analysis: QueryAnalysis, query: str, history: str = "") -> QueryAnalysis:
    text = query.lower()

    # 1. Order IDs follow a fixed pattern, so a regex is more reliable than the LLM.
    #    Use the ID in the latest message; otherwise the most recent one from the chat history
    #    (for follow-ups like "and when will it arrive?"). Anything else was invented by the model.
    in_query = ORDER_ID_RE.findall(query)
    in_history = ORDER_ID_RE.findall(history or "")
    if in_query:
        analysis.order_id = f"ORD{in_query[0]}"
    elif in_history:
        analysis.order_id = f"ORD{in_history[-1]}"
    else:
        analysis.order_id = None

    # 2. Only escalate when there is a real reason
    has_trigger = any(word in text for word in ESCALATION_WORDS)
    if analysis.needs_human and not has_trigger and analysis.sentiment != "angry":
        analysis.needs_human = False
    if has_trigger and analysis.sentiment in ("negative", "angry"):
        analysis.needs_human = True

    # 3. Calm, routine questions are not high urgency
    if analysis.urgency == "high" and analysis.sentiment in ("neutral", "positive") and not has_trigger:
        analysis.urgency = "medium" if analysis.order_id else "low"

    return analysis


# LLM classification followed by guardrails, still a single chain
analysis_chain = RunnableLambda(
    lambda x: apply_guardrails(llm_analysis_chain.invoke(x), x["query"], x.get("history", ""))
)


if __name__ == "__main__":
    samples = [
        "Where is my order ORD1002? It's been a week.",
        "This is the THIRD time my refund failed. Rs 5,999 is gone and nobody helps. I'll go to consumer court!",
        "Can I change the delivery address for ORD1003 to 21 Park Street, Kolkata?",
        "What are your support hours?",
        "I can't log in, the OTP never arrives.",
    ]
    for msg in samples:
        result: QueryAnalysis = analysis_chain.invoke({"query": msg, "history": "(none)"})
        print(f"\nMessage: {msg}")
        print(result.model_dump_json(indent=2))
