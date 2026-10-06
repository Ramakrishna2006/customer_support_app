import step3_react_agent as agent


class ScriptedLLM:
    """Stands in for the model: returns pre-written ReAct text, one reply per call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, inputs):
        self.prompts.append(inputs)
        return self.replies.pop(0)


def test_parser_reads_the_newest_step_when_the_model_echoes():
    echoed = (
        "Question: change address on ORD1003\n"
        "Thought: check first\nAction: check_order_status\nAction Input: {\"order_id\": \"ORD1003\"}\n\n"
        "Thought: It is Processing.\nAction: update_delivery_address\n"
        "Action Input: {\"order_id\": \"ORD1003\", \"new_address\": \"Kolkata\"}"
    )
    parsed = agent.parse_react_output(echoed)
    assert parsed["type"] == "action"
    assert parsed["tool"] == "update_delivery_address"
    assert parsed["thought"] == "It is Processing."


def test_parser_final_answer():
    parsed = agent.parse_react_output("Thought: I now know the final answer\nFinal Answer: It ships today.")
    assert parsed == {"type": "final", "thought": "I now know the final answer", "answer": "It ships today.",
                      "raw": "Thought: I now know the final answer\nFinal Answer: It ships today."}


def test_parser_invalid():
    assert agent.parse_react_output("hello there")["type"] == "invalid"


def test_blocks_made_up_order_ids():
    assert agent._unmentioned_order_id({"order_id": "ORD1002"}, "Do you offer EMI?") == "ORD1002"
    assert agent._unmentioned_order_id({"order_id": "ord-1003"}, "change ORD 1003 please") is None
    assert agent._unmentioned_order_id({"query": "emi"}, "anything") is None


def test_full_loop_uses_tools_then_answers(temp_data, monkeypatch):
    llm = ScriptedLLM(
        'Thought: look it up\nAction: check_order_status\nAction Input: {"order_id": "ORD1004"}',
        "Thought: I now know the final answer\nFinal Answer: ORD1004 is delayed.",
    )
    monkeypatch.setattr(agent, "react_chain", llm)
    result = agent.run_react_agent("Where is ORD1004?")
    assert result["answer"] == "ORD1004 is delayed."
    assert [s["tool"] for s in result["steps"]] == ["check_order_status"]
    assert "Delayed" in result["steps"][0]["observation"]


def test_repeated_calls_are_blocked_and_fallback_answers(temp_data, monkeypatch):
    stuck = 'Thought: check\nAction: check_order_status\nAction Input: {"order_id": "ORD1004"}'
    monkeypatch.setattr(agent, "react_chain", ScriptedLLM(stuck, stuck, stuck, stuck, stuck))

    class FakeFinal:
        def invoke(self, inputs):
            return "FALLBACK: " + inputs["observations"]

    monkeypatch.setattr(agent, "final_answer_chain", FakeFinal())
    result = agent.run_react_agent("Where is ORD1004?")
    assert len(result["steps"]) == 1  # the tool really ran only once
    assert result["answer"].startswith("FALLBACK") and "Delayed" in result["answer"]


def test_guessed_order_id_never_reaches_the_tool(temp_data, monkeypatch):
    llm = ScriptedLLM(
        'Thought: check\nAction: check_order_status\nAction Input: {"order_id": "ORD1002"}',
        "Thought: I now know the final answer\nFinal Answer: Please share your order ID.",
    )
    monkeypatch.setattr(agent, "react_chain", llm)
    result = agent.run_react_agent("Do you offer EMI?")
    assert result["steps"] == []
    assert "did not mention order ORD1002" in llm.prompts[1]["scratchpad"]
