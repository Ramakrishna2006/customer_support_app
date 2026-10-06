# Chapter 2: Your first chain

**Goal:** understand the three building blocks every LangChain app is made of, and the three ways to run them.

## 2.1 The idea

A **chain** passes data through a series of steps, where each step's output becomes the next step's input:

```
{"company": ..., "query": ...}  →  Prompt template  →  messages  →  LLM  →  AI message  →  Parser  →  "plain text"
```

LangChain Expression Language (**LCEL**) connects the steps with the `|` operator, like a Unix pipe:

```python
chain = prompt | llm | parser
```

Every piece (prompt, model, parser, and later retrievers, tools, and your own functions) is a **Runnable**.
All Runnables share the same methods, which is why they plug together.

## 2.2 The code

Create `step1_basic_chain.py`:

````python
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
````

What each part does:

- **`ChatPromptTemplate.from_messages`** builds a list of chat messages with `{placeholders}`. The **system**
  message sets the assistant's role and rules; the **human** message carries the customer's question.
- **The LLM** receives the messages and returns an `AIMessage` object.
- **`StrOutputParser`** pulls the text out of that object.

The three ways to run any Runnable:

| Method | Use it for |
|---|---|
| `.invoke(input)` | One input, one output |
| `.batch([inputs])` | Many inputs at once (runs in parallel threads) |
| `.stream(input)` | Get the answer piece by piece as it's generated (live chat feel) |

## 2.3 Run it

```powershell
python step1_basic_chain.py
```

## Checkpoint

You should see a single reply, two batch replies, and a streamed reply printing word by word.

**Look closely at the first reply.** The prompt says *never invent order details*, yet the model may say
something like *"I've checked your order and it's still being processed."* It checked nothing: it has no
access to any order data. This is **hallucination**, and it's the problem the rest of the project solves.
Chapter 5 gives the model real data through tools.

> On Ollama, `.batch()` can look frozen for a minute: a CPU handles the requests one after another. Don't press
> Ctrl+C; a `KeyboardInterrupt` traceback means *you* stopped it, not that it crashed.

## Try it yourself

1. Add a `{tone}` placeholder ("formal" or "casual") and pass it in `.invoke()`.
2. Set `temperature=1` in `get_llm()` and run twice. Then `temperature=0`. Compare.
3. Make the system message reply in Hindi or Telugu.

**Next: [Chapter 3: Structured output →](03-structured-output.md)**
