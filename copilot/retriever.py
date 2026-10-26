# Vector search over the docs, for the agent. Same chunking as rag_minimal.py (imported, not copied),
# but the index is FAISS instead of a raw numpy matrix, and the embeddings are cached on disk.
# Why cache: embedding all chunks with Ollama takes seconds; the docs rarely change, so we embed
# once and reuse the vectors until any doc (or the embedding model) changes.

import hashlib
from pathlib import Path

import faiss
import numpy as np

from rag_minimal import DOCS, DOCS_DIR, chunk_markdown

INDEX_DIR = Path(__file__).parent.parent / "index"


class Retriever:
    def __init__(self, llm, docs_dir: Path = DOCS_DIR, docs: list[str] = DOCS, cache_dir: Path = INDEX_DIR):
        self.llm = llm
        texts = {name: (docs_dir / name).read_text(encoding="utf-8") for name in docs}
        self.chunks = [c for name, text in texts.items() for c in chunk_markdown(text, name)]

        # The cache key is a hash of the embedding model + every CHUNK's text, so changing a doc, the chunk
        # size or the chunking code rebuilds the index. (A first version hashed only the raw docs: after a
        # chunking change it silently kept serving the old vectors.)
        key = hashlib.sha256((llm.embed_model + "".join(c["text"] for c in self.chunks)).encode()).hexdigest()[:16]
        cache = cache_dir / f"vectors-{key}.npy"
        if cache.exists():
            vectors = np.load(cache)
        else:
            vectors = llm.embed([f"search_document: {c['text']}" for c in self.chunks])
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache, vectors)

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)  # unit length, so inner product == cosine similarity
        # IndexFlatIP = exact search by inner product (no approximation). With ~200 chunks an
        # approximate index (IVF, HNSW) would add complexity for no speed gain.
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, question: str, k: int = 4) -> list[dict]:
        q = np.ascontiguousarray(self.llm.embed([f"search_query: {question}"]), dtype=np.float32)
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, k)  # both shaped (1, k): one row per query
        return [{**self.chunks[i], "score": round(float(s), 4)}
                for s, i in zip(scores[0], ids[0]) if i != -1]
