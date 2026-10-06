"""
STEP 3: ReAct prompting (Reason + Act).

Some questions can't be answered from the FAQ alone. "Where is ORD1002?" needs
real data. A ReAct agent solves this by looping:

    Thought      -> the model reasons about what to do next
    Action       -> it picks a tool
    Action Input -> it says what to pass to the tool
    Observation  -> WE run the tool and paste the result back in
    ... repeat until ...
    Final Answer -> the reply for the customer

Part A builds this loop by hand so you can see exactly how ReAct prompting works.
Part B shows LangChain's built-in agent, which does the same job with native
tool calling. Use Part A to learn; Part B is what you'd usually use in production.
"""
import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import COMPANY_NAME, get_llm
from tools import ALL_TOOLS

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
TOOL_LIST = "\n".join(f"- {t.name}{list(t.args)}: {' '.join(t.description.split())}" for t in ALL_TOOLS)

# ---------------------------------------------------------------------------
# PART A: ReAct prompting from scratch
# ---------------------------------------------------------------------------
REACT_SYSTEM = """You are a helpful customer support agent for {company}.
Answer the customer's question. You have access to the following tools:

{tools}

Work step by step using EXACTLY this format:

Thought: reason about what to do next
Action: the tool to use, exactly one of [{tool_names}]
Action Input: the tool's arguments as a JSON object

Then STOP. The system runs the tool and shows you the result as an Observation.
When you have enough information, write instead:

Thought: I now know the final answer
Final Answer: a friendly, concise reply to the customer

Rules:
- Write ONLY the next step. Never repeat the question or steps you have already taken.
- Never call the same tool with the same input twice. Use the Observation you already have.
- Never invent order details. Always look them up with a tool.
- Start the Final Answer by directly answering what the customer asked. For "where is my order",
  give the status, expected delivery date and tracking number.
- Only mention policies that are relevant to the customer's question.
- Check company policy with search_knowledge_base before promising refunds or returns.
- If you cannot solve the problem, create a support ticket and tell the customer the ticket ID.
- If you need an order ID the customer has not given, ask for it in your Final Answer."""

REACT_HUMAN = """Conversation so far:
{history}

Question: {query}

Steps already taken:
{scratchpad}

Write the next step."""

react_prompt = ChatPromptTemplate.from_messages([("system", REACT_SYSTEM), ("human", REACT_HUMAN)]).partial(
    company=COMPANY_NAME, tools=TOOL_LIST, tool_names=", ".join(TOOLS_BY_NAME)
)

llm = get_llm(temperature=0)

# stop= makes the model halt before it can hallucinate its own "Observation:"
react_chain = react_prompt | llm.bind(stop=["\nObservation:", "Observation:"]) | StrOutputParser()

# If the loop runs out of steps, this chain writes a reply from whatever the tools found
final_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are a customer support agent for {COMPANY_NAME}. Answer EXACTLY what the customer "
            "asked, using ONLY the tool results below. Start by directly answering their question "
            "(for 'where is my order': status, expected delivery date and tracking number). Ignore "
            "tool results that are not relevant to their question. Write a friendly, concise reply "
            "(under 100 words). Never invent details.",
        ),
        ("human", "Customer question: {query}\n\nTool results:\n{observations}"),
    ]
)
final_answer_chain = final_answer_prompt | llm | StrOutputParser()

KEYWORDS = r"(?:Thought|Action|Action\s*Input|Observation|Question|Final\s*Answer)\s*:"
ACTION_RE = re.compile(
    r"Action\s*:\s*([^\n]+?)\s*\n\s*Action\s*Input\s*:\s*(.*?)(?=\n\s*" + KEYWORDS + r"|\Z)",
    re.DOTALL,
)


def parse_react_output(text: str) -> dict:
    """Turn the model's raw text into either a tool call or a final answer.

    Small models often echo earlier steps before writing a new one, so we read
    the LAST Action / Final Answer in the text, which is the model's newest step.
    """
    text = text.split("Observation:")[0].strip()  # anything after this was hallucinated
    actions = list(ACTION_RE.finditer(text))
    last_action = actions[-1] if actions else None
    final_pos = text.rfind("Final Answer:")

    if final_pos != -1 and (last_action is None or final_pos > last_action.start()):
        before = text[:final_pos]
        thought = before.rsplit("Thought:", 1)[-1].strip() if "Thought:" in before else ""
        answer = text[final_pos + len("Final Answer:"):].strip()
        return {"type": "final", "thought": thought, "answer": answer, "raw": text}

    if last_action:
        before = text[: last_action.start()]
        thought = before.rsplit("Thought:", 1)[-1].strip() if "Thought:" in before else ""
        tool_input = re.sub(r"^```(?:json)?|```$", "", last_action.group(2).strip()).strip()
        return {
            "type": "action",
            "thought": thought,
            "tool": last_action.group(1).strip().strip("`'\""),
            "tool_input": tool_input,
            "raw": text,
        }
    return {"type": "invalid", "raw": text}


def parse_tool_input(raw: str, tool) -> dict:
    """Accept JSON (preferred) or, for one-argument tools, a plain string."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    first_arg = next(iter(tool.args))
    return {first_arg: raw.strip().strip('"').strip("'")}


def _call_key(tool_name: str, args: dict) -> str:
    """A normalised fingerprint of a tool call, used to spot repeats."""
    return tool_name + json.dumps({k: str(v).strip().upper() for k, v in args.items()}, sort_keys=True)


def _normalise_ids(text: str) -> set[str]:
    return {f"ORD{n}" for n in re.findall(r"ORD[\s\-#]*(\d+)", text, re.IGNORECASE)}


def _unmentioned_order_id(args: dict, conversation: str) -> str | None:
    """Return the order ID if the model is using one the customer never wrote, else None."""
    if "order_id" not in args:
        return None
    raw = re.sub(r"[^A-Za-z0-9]", "", str(args["order_id"])).upper()
    order_id = f"ORD{raw}" if raw.isdigit() else raw
    return None if order_id in _normalise_ids(conversation) else order_id


def run_react_agent(query: str, history: str = "(none)", max_steps: int = 4, verbose: bool = False) -> dict:
    """Run the Thought -> Action -> Observation loop until a Final Answer."""
    scratchpad = ""
    steps = []
    done_calls: dict[str, str] = {}  # call fingerprint -> observation
    repeats = 0

    for _ in range(max_steps):
        output = react_chain.invoke({"query": query, "history": history, "scratchpad": scratchpad or "(none yet)"})
        parsed = parse_react_output(output)

        if parsed["type"] == "final":
            if verbose:
                print(f"Thought: {parsed['thought']}\nFinal Answer: {parsed['answer']}\n")
            return {"answer": parsed["answer"], "steps": steps}

        if parsed["type"] == "invalid":
            if verbose:
                print(f"(invalid format)\n{parsed['raw'][:300]}\n")
            # Remind the model of the format and let it try again
            scratchpad += (
                "Observation: Invalid format. Reply with 'Thought:', 'Action:' and 'Action Input:', "
                "or with 'Final Answer:'.\n"
            )
            continue

        # Only the NEW step goes into the scratchpad, never the model's echoed text
        step_text = f"Thought: {parsed['thought']}\nAction: {parsed['tool']}\nAction Input: {parsed['tool_input']}"
        if verbose:
            print(step_text)

        tool = TOOLS_BY_NAME.get(parsed["tool"])
        if tool is None:
            observation = f"Unknown tool '{parsed['tool']}'. Choose one of: {', '.join(TOOLS_BY_NAME)}."
        else:
            args = parse_tool_input(parsed["tool_input"], tool)
            key = _call_key(tool.name, args)
            bad_id = _unmentioned_order_id(args, f"{query}\n{history}")
            if bad_id:
                # Guardrail: block lookups/changes on orders the customer never mentioned
                observation = (
                    f"The customer did not mention order {bad_id}. Never guess order IDs. "
                    "If you need one, ask the customer for it in your Final Answer."
                )
            elif key in done_calls:
                # Don't run it again: repeating a tool like create_support_ticket would create duplicates
                repeats += 1
                observation = (
                    f"You already called {tool.name} with this input. The result was: {done_calls[key]} "
                    "Do not repeat it. Write the Final Answer now, or use a different tool."
                )
                if repeats >= 1:
                    # The model is going round in circles. Asking again wastes a slow LLM call,
                    # so go straight to the fallback answer below.
                    if verbose:
                        print(step_text)
                    break
            else:
                try:
                    observation = str(tool.invoke(args))
                except Exception as exc:  # a broken tool call should not crash the chat
                    observation = f"Tool error: {exc}"
                done_calls[key] = observation
                steps.append(
                    {"thought": parsed["thought"], "tool": parsed["tool"], "tool_input": parsed["tool_input"], "observation": observation}
                )
        if verbose:
            print(f"Observation: {observation}\n")
        scratchpad += f"{step_text}\nObservation: {observation}\n"

    # Fallback: the model never wrote a Final Answer, so write one from the tool results
    if steps:
        observations = "\n".join(f"- {s['tool']}: {s['observation']}" for s in steps)
        answer = final_answer_chain.invoke({"query": query, "observations": observations})
        if verbose:
            print(f"(loop stopped; writing the answer from tool results)\nFinal Answer: {answer}\n")
        return {"answer": answer, "steps": steps}

    return {
        "answer": "I'm sorry, I couldn't resolve this automatically. A human agent will follow up with you shortly.",
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# PART B: LangChain's built-in agent (native tool calling, same ReAct idea)
# ---------------------------------------------------------------------------
def build_tool_calling_agent():
    from langchain.agents import create_agent

    return create_agent(
        model=get_llm(temperature=0),
        tools=ALL_TOOLS,
        system_prompt=(
            f"You are a helpful customer support agent for {COMPANY_NAME}. Never invent order details; "
            "use tools to look them up. Check policy before promising refunds or changes."
        ),
    )


if __name__ == "__main__":
    questions = [
        "Where is my order ORD1004? It should have arrived last week.",
        "Please change the address on ORD1003 to 21 Park Street, Kolkata.",
        "My order ORD1001 arrived broken. Can I return it and how long will the refund take on UPI?",
    ]

    print("=" * 70 + "\nPART A: Hand-built ReAct loop\n" + "=" * 70)
    for q in questions:
        print(f"\nQuestion: {q}\n")
        result = run_react_agent(q, verbose=True)
        print(f">>> REPLY TO CUSTOMER: {result['answer']}\n" + "-" * 70)

    print("\n" + "=" * 70 + "\nPART B: LangChain create_agent\n" + "=" * 70)
    agent = build_tool_calling_agent()
    state = agent.invoke({"messages": [{"role": "user", "content": questions[0]}]})
    for message in state["messages"]:
        message.pretty_print()
