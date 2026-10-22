"""Retrieval over your own notes and PDFs.

The pipeline is the standard one: read each document, split it into overlapping
chunks, turn each chunk into a vector with a local embedding model, and store the
vectors in a Chroma collection on disk. At query time we embed the question the
same way and ask Chroma for the nearest chunks — those are the passages the agent
answers from, and each carries its source so answers can be cited.

Embeddings run locally via sentence-transformers (no API, no key). The model and
Chroma client are loaded lazily so importing this module stays cheap.
"""

import os

EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast, local sentence-transformers model
CHROMA_DIR = os.environ.get("MINI_MUSE_CHROMA", "chroma_db")
COLLECTION = "docs"

CHUNK_SIZE = 800     # characters per chunk
CHUNK_OVERLAP = 100  # characters shared between neighbours, so a fact split across a boundary still lands whole somewhere

_EMBEDDER = None
_CLIENT = None


def _embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(EMBED_MODEL)  # downloads ~90MB on first use
    return _EMBEDDER


def chroma_client():
    """The shared on-disk Chroma client, so other modules reuse one connection."""
    global _CLIENT
    if _CLIENT is None:
        import chromadb
        _CLIENT = chromadb.PersistentClient(path=CHROMA_DIR)
    return _CLIENT


def embedder():
    """The shared local embedding model (loaded once, reused everywhere)."""
    return _embedder()


def _collection():
    return chroma_client().get_or_create_collection(COLLECTION)


def _read_doc(path: str) -> str:
    """Pull plain text out of a .pdf, .md, or .txt file."""
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def _chunk(text: str) -> list[str]:
    """Slice text into overlapping fixed-size chunks."""
    text = text.strip()
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + CHUNK_SIZE])
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def ingest(folder: str) -> int:
    """Index every .md/.txt/.pdf under `folder`. Returns the chunk count added."""
    model = _embedder()
    col = _collection()

    total = 0
    for root, _, files in os.walk(folder):
        for name in files:
            if not name.lower().endswith((".md", ".txt", ".pdf")):
                continue
            path = os.path.join(root, name)
            chunks = _chunk(_read_doc(path))
            if not chunks:
                continue
            embeddings = model.encode(chunks).tolist()
            # upsert keyed by path+chunk index, so re-ingesting updates in place
            # instead of duplicating.
            col.upsert(
                ids=[f"{path}::{i}" for i in range(len(chunks))],
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{"source": path, "chunk": i} for i in range(len(chunks))],
            )
            total += len(chunks)
    return total


def search(query: str, k: int = 5) -> list[dict]:
    """Return the k chunks nearest to `query`, each with its source for citation."""
    col = _collection()
    if col.count() == 0:
        return []

    query_embedding = _embedder().encode([query]).tolist()
    res = col.query(query_embeddings=query_embedding, n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [
        {"text": d, "source": m["source"], "chunk": m["chunk"], "distance": dist}
        for d, m, dist in zip(docs, metas, dists)
    ]
