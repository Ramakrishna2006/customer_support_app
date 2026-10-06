"""
STEP 1: Your first LangChain chain.

A chain links components so the output of one becomes the input of the next:

    Prompt Template  ->  LLM  ->  Output Parser

LangChain Expression Language (LCEL) uses the | (pipe) operator to connect them.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import get_llm

# 1. PROMPT TEMPLATE: a reusable prompt with placeholders in {curly braces}
support_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a friendly customer support agent for {company}. "
            "Answer clearly in 3-4 sentences. If you don't know something, "
            "say a human agent will follow up. Never invent order details.",
        ),
        ("human", "Customer name: {customer_name}\nQuery: {query}"),
    ]
)

# 2. LLM: the model that generates the answer
llm = get_llm()

# 3. OUTPUT PARSER: turns the model's message object into a plain string
parser = StrOutputParser()

# 4. THE CHAIN: connect the pieces with |
support_chain = support_prompt | llm | parser


if __name__ == "__main__":
    # .invoke() runs the chain once with the given inputs
    reply = support_chain.invoke(
        {
            "company": "ShopEasy",
            "customer_name": "Ravi",
            "query": "My order was supposed to arrive yesterday but it hasn't come. What should I do?",
        }
    )
    print("=== Single reply ===")
    print(reply)

    # .batch() runs the chain on many inputs at once
    queries = [
        "How do I return a damaged product?",
        "Can I change my delivery address after ordering?",
    ]
    replies = support_chain.batch(
        [{"company": "ShopEasy", "customer_name": "Priya", "query": q} for q in queries]
    )
    print("\n=== Batch replies ===")
    for q, r in zip(queries, replies):
        print(f"\nQ: {q}\nA: {r}")

    # .stream() prints the answer word by word, like a live chat
    print("\n=== Streaming reply ===")
    for chunk in support_chain.stream(
        {"company": "ShopEasy", "customer_name": "Anil", "query": "What are your support hours?"}
    ):
        print(chunk, end="", flush=True)
    print()
