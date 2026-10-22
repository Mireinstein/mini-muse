"""Long-term memory — facts the agent keeps across conversations.

Where RAG holds the user's documents, this is the agent's own memory: short
facts it saves with `remember` and pulls back semantically with `recall`. This
is what lets Mini Muse know your preferences (a dietary restriction, how you
like things done) in a later, unrelated conversation.

Memories are embedded with the same local model as rag.py and stored in a
separate Chroma collection, so there's no API or key involved and recall is by
meaning rather than keyword.
"""

import time
import uuid

from . import rag

MEMORY_COLLECTION = "memories"


def _collection():
    # Reuse rag's embedder and Chroma client; a separate collection keeps the
    # agent's memories apart from the user's indexed documents.
    return rag.chroma_client().get_or_create_collection(MEMORY_COLLECTION)


def remember(text: str) -> str:
    """Store one fact. Returns its id (so it can be forgotten later)."""
    text = text.strip()
    if not text:
        return ""
    mem_id = uuid.uuid4().hex
    embedding = rag.embedder().encode([text]).tolist()
    _collection().upsert(
        ids=[mem_id],
        embeddings=embedding,
        documents=[text],
        metadatas=[{"created_at": time.time()}],
    )
    return mem_id


def recall(query: str, k: int = 5) -> list[str]:
    """Return the facts most relevant to `query`, nearest first."""
    col = _collection()
    count = col.count()
    if count == 0:
        return []
    embedding = rag.embedder().encode([query]).tolist()
    res = col.query(query_embeddings=embedding, n_results=min(k, count))
    return res["documents"][0]


def all_memories() -> list[dict]:
    """Every stored memory with its id — powers a 'what do you know about me' view."""
    col = _collection()
    if col.count() == 0:
        return []
    res = col.get()
    return [{"id": i, "text": d} for i, d in zip(res["ids"], res["documents"])]


def forget(mem_id: str) -> bool:
    """Delete one memory by id. Returns False if it wasn't there."""
    col = _collection()
    existing = set(col.get()["ids"])
    if mem_id not in existing:
        return False
    col.delete(ids=[mem_id])
    return True
