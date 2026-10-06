# Chapter 4: Knowledge base (RAG)

**Goal:** answer policy questions from *your company's* documents instead of the model's general knowledge.

## 4.1 Why RAG

Ask a plain LLM *"How long does a UPI refund take?"* and it will make up a believable number. Your company's
real answer is in its help centre. **Retrieval-Augmented Generation (RAG)** fixes this in two steps:

1. **Retrieve:** find the few help-centre passages most relevant to the question.
2. **Generate:** give those passages to the LLM and tell it to answer **only** from them.

## 4.2 How retrieval works

```
FAQ document ──split──► 9 chunks ──embed──► 9 vectors ──► vector store
                                                              │
question ──────────────embed──► 1 vector ──similarity search──┘──► top 3 chunks
```

- **Chunking:** split the document into pieces small enough to be relevant on their own. Our FAQ has one
  topic per `##` heading, so each section is one chunk. (For long documents, use a text splitter.)
- **Embedding:** an embedding model turns text into a list of numbers (a vector) that represents its
  *meaning*. Texts with similar meaning get similar vectors.
- **Similarity search:** the question is embedded the same way, and the store returns the chunks whose vectors
  are closest. *"UPI refund"* finds the **Refunds** section even if the words don't match exactly.

## 4.3 The help-centre document

Create `data/faq.md`:

````markdown
# ShopEasy Help Centre

## Return policy
Most products can be returned within 30 days of delivery if they are unused and in their original packaging. Damaged or defective items can be returned within 30 days even if opened. Items marked "non-returnable" on the product page, such as innerwear and personal care products, cannot be returned. To start a return, go to My Orders, choose the item and tap "Return". A pickup is scheduled within 2 working days.

## Refunds
Refunds are processed after the returned item passes a quality check, usually within 2 days of pickup. Money goes back to the original payment method. UPI and wallet refunds arrive in 1-3 working days, card refunds in 5-7 working days, and net banking refunds in 3-5 working days. Cash-on-delivery orders are refunded to your ShopEasy wallet or a bank account you provide.

## Shipping and delivery
Standard delivery takes 3-7 working days and is free on orders above Rs 499. Express delivery (1-2 days) costs Rs 99 and is available in major cities. You can track any shipped order from My Orders using the tracking number. If an order is marked "Delayed", it usually arrives within 3 extra working days and you will be notified of the new date.

## Changing the delivery address
The delivery address can be changed only while the order status is "Processing". Once an order has shipped the address cannot be changed, but you can contact the courier with your tracking number to request a hold at their nearest centre.

## Order cancellation
Orders can be cancelled free of charge any time before they ship. After shipping, you can refuse the delivery or place a return once it arrives. Refunds for cancelled prepaid orders follow the refund timelines above.

## Payment methods
We accept UPI, credit and debit cards, net banking, ShopEasy wallet, EMI on cards above Rs 3,000, and cash on delivery for orders up to Rs 10,000. If money was deducted but the order failed, it is automatically refunded within 5 working days.

## Warranty
Electronics carry a 1-year manufacturer warranty unless stated otherwise on the product page. For warranty claims, contact the brand's service centre with your invoice, which you can download from My Orders. ShopEasy support can help you find the nearest authorised service centre.

## Account and login
To reset your password, tap "Forgot password" on the login screen and follow the OTP sent to your registered mobile number. If you no longer have access to that number, contact support with your registered email address to verify your identity. We will never ask for your password, OTP or card PIN.

## Support hours and contact
Chat support is available 24/7. Phone support (1800-123-4567, toll free) is open 9 AM to 9 PM IST, Monday to Saturday. Email support at help@shopeasy.example replies within 24 hours. Urgent issues such as payment failures or fraud are prioritised.
````

## 4.4 The code

Create `knowledge_base.py`:

````python
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
````

Notes:

- **`InMemoryVectorStore`** keeps vectors in RAM. It's perfect for 9 chunks and rebuilt in a second at
  start-up. For thousands of documents, use FAISS or Chroma, which save the index to disk.
- **`@lru_cache`** builds the store once and reuses it.
- Each chunk **starts with its section title**, so the app can show which sections it used ("sources").

## 4.5 Run it

```powershell
python knowledge_base.py
```

## Checkpoint

```
Loaded 9 FAQ sections.

Q: How long does a UPI refund take?
   -> Refunds
   -> Payment methods

Q: Can I change my address after ordering?
   -> Changing the delivery address
   -> Order cancellation
```

The first result for each question should be the obviously right section. If you get
`model "nomic-embed-text" not found`, run `ollama pull nomic-embed-text`.

The *generation* half of RAG (answering from these chunks) is in chapter 7's FAQ branch, where you'll also
see how to make a small model copy facts exactly.

**Next: [Chapter 5: Tools →](05-tools.md)**
