from types import SimpleNamespace

import step4_dag_workflow as dag


def state(query="question", **analysis):
    fields = dict(intent="general_question", sentiment="neutral", urgency="low", order_id=None, needs_human=False)
    fields.update(analysis)
    return {"query": query, "analysis": SimpleNamespace(**fields)}


def test_escalation():
    assert dag.should_escalate(state(needs_human=True))
    assert dag.should_escalate(state(sentiment="angry", urgency="high"))
    assert not dag.should_escalate(state(sentiment="angry", urgency="medium"))


def test_quick_lookup_for_status_questions():
    assert dag.should_quick_lookup(state("Where is my order ORD1002?", order_id="ORD1002"))


def test_agent_for_action_requests():
    s = state("Please change the address on ORD1003", order_id="ORD1003")
    assert not dag.should_quick_lookup(s)
    assert dag.should_use_agent(s)


def test_general_questions_go_to_faq_even_if_mislabelled():
    s = state("Do you offer EMI?", intent="order_status")  # small models often mislabel this
    assert not dag.should_quick_lookup(s)
    assert not dag.should_use_agent(s)
