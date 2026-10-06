from step2_structured_output import QueryAnalysis, apply_guardrails


def make(**overrides) -> QueryAnalysis:
    fields = dict(intent="order_status", sentiment="neutral", urgency="high", order_id=None,
                  needs_human=True, summary="test")
    fields.update(overrides)
    return QueryAnalysis(**fields)


def test_fixes_a_missed_order_id_and_false_escalation():
    a = apply_guardrails(make(), "Where is my order ORD1002? It's been a week.")
    assert a.order_id == "ORD1002"
    assert a.needs_human is False
    assert a.urgency == "medium"


def test_removes_invented_order_id():
    a = apply_guardrails(make(intent="general_question", order_id="ORD555"), "What are your support hours?")
    assert a.order_id is None
    assert a.urgency == "low"


def test_order_id_from_history_for_follow_ups():
    a = apply_guardrails(make(needs_human=False), "When will it arrive?", "Customer: where is ORD1006?")
    assert a.order_id == "ORD1006"


def test_escalates_legal_threats():
    a = apply_guardrails(
        make(sentiment="negative", needs_human=False),
        "This is the THIRD time my refund failed. I'll go to consumer court!",
    )
    assert a.needs_human is True


def test_keeps_explicit_request_for_a_human():
    a = apply_guardrails(make(urgency="medium"), "Please connect me to a human agent")
    assert a.needs_human is True
