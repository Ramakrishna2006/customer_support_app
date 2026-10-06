from knowledge_base import format_docs, load_faq_documents


def test_faq_is_split_into_sections():
    docs = load_faq_documents()
    sections = [d.metadata["section"] for d in docs]
    assert len(docs) == 9
    assert sections[0] == "Return policy" and "Payment methods" in sections


def test_each_chunk_starts_with_its_title():
    docs = load_faq_documents()[:2]
    assert format_docs(docs).startswith("Return policy\n")
