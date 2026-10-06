"""
Knowledge base for Retrieval-Augmented Generation (RAG).

The help-centre FAQ (data/faq.md) is split into one chunk per "## " section,
turned into embeddings (vectors), and stored in an in-memory vector store.
When a customer asks something, we find the most similar sections and give
them to the LLM as context, so answers come from company policy, not guesswork.
"""
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from config import DATA_DIR, get_embeddings

FAQ_PATH = DATA_DIR / "faq.md"


def load_faq_documents() -> list[Document]:
    """Split the FAQ markdown into one Document per section."""
    text = FAQ_PATH.read_text(encoding="utf-8")
    docs = []
    for section in text.split("\n## ")[1:]:  # skip the page title before the first section
        title, _, body = section.partition("\n")
        docs.append(
            Document(
                page_content=f"{title.strip()}\n{body.strip()}",
                metadata={"section": title.strip(), "source": FAQ_PATH.name},
            )
        )
    return docs


@lru_cache(maxsize=1)
def get_vector_store() -> InMemoryVectorStore:
    """Embed the FAQ once and cache the store for the rest of the session."""
    return InMemoryVectorStore.from_documents(load_faq_documents(), get_embeddings())


def get_retriever(k: int = 3):
    """A retriever returns the k most relevant FAQ sections for a query."""
    return get_vector_store().as_retriever(search_kwargs={"k": k})


def format_docs(docs: list[Document]) -> str:
    """Join retrieved sections into one block of text for the prompt."""
    return "\n\n".join(d.page_content for d in docs)  # each chunk already starts with its section title


def search_faq(query: str, k: int = 3) -> str:
    return format_docs(get_retriever(k).invoke(query))


if __name__ == "__main__":
    print(f"Loaded {len(load_faq_documents())} FAQ sections.\n")
    for q in ["How long does a UPI refund take?", "Can I change my address after ordering?"]:
        print(f"Q: {q}")
        for doc in get_retriever(k=2).invoke(q):
            print("   ->", doc.metadata["section"])
        print()
