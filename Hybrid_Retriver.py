"""
hybrid_retriever.py

Retrieval pipeline for the Legal Assistant Bot:

    question
       ├─► dense search  (FAISS + MiniLM embeddings)  ─┐
       └─► keyword search (BM25)                       ─┴─► RRF fusion ─► cross-encoder rerank ─► top-k

Why each stage exists
---------------------
* Dense search understands meaning ("punishment for stealing" ~ "penalty for theft")
  but can miss exact identifiers such as "Section 420" or "Article 21".
* BM25 matches exact words/numbers but has no idea about synonyms.
* RRF (Reciprocal Rank Fusion) merges the two ranked lists using only RANKS,
  so we never have to compare incomparable scores (L2 distance vs BM25 score).
* A cross-encoder reads (question, chunk) TOGETHER, which is far more accurate
  than comparing two independently-made embeddings, but too slow to run on
  every chunk, so we run it only on the small fused shortlist.
"""

import re
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    """Lowercase + split into word tokens. Keeps numbers, so 'Section 420' -> ['section', '420']."""
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    def __init__(
        self,
        vectorstore,
        top_k: int = 4,          # chunks finally sent to the LLM
        fetch_k: int = 20,       # candidates pulled from EACH retriever
        rerank_pool: int = 12,   # fused candidates passed to the cross-encoder
        use_bm25: bool = True,
        use_rerank: bool = True,
        rrf_k: int = 60,         # standard RRF constant
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.rerank_pool = rerank_pool
        self.rrf_k = rrf_k

        # All chunks live inside the saved FAISS docstore, so there is no need to
        # re-run ingest.py or save a second file for BM25.
        self.docs: List[Document] = list(vectorstore.docstore._dict.values())

        # ---- sparse (keyword) index, built in memory at startup -----------------
        self.bm25 = None
        if use_bm25 and self.docs:
            self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs])

        # ---- cross-encoder reranker ------------------------------------------------
        self.reranker = None
        if use_rerank:
            try:
                from sentence_transformers import CrossEncoder

                self.reranker = CrossEncoder(reranker_model)
            except Exception as e:  # no internet / model missing -> degrade gracefully
                print(f"[HybridRetriever] Reranker disabled ({e})")

        # Cache results per question: the app calls retrieve() twice for the same
        # question (once in the chain, once for the "sources" panel).
        self._cached = lru_cache(maxsize=256)(self._retrieve)

    # ------------------------------------------------------------------ stages
    def _dense(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.fetch_k)

    def _sparse(self, query: str) -> List[Document]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        top_idx = np.argsort(scores)[::-1][: self.fetch_k]
        # score <= 0 means "no keyword overlap" -> don't pollute the fusion
        return [self.docs[i] for i in top_idx if scores[i] > 0]

    def _rrf(self, rankings: Sequence[Sequence[Document]]) -> List[Document]:
        """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (rrf_k + rank)."""
        fused: Dict[str, Tuple[Document, float]] = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking, start=1):
                key = doc.page_content
                prev_doc, prev_score = fused.get(key, (doc, 0.0))
                fused[key] = (prev_doc, prev_score + 1.0 / (self.rrf_k + rank))
        ordered = sorted(fused.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ordered]

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if self.reranker is None or not docs:
            return docs
        scores = self.reranker.predict([(query, d.page_content) for d in docs])
        order = np.argsort(scores)[::-1]
        return [docs[i] for i in order]

    # ------------------------------------------------------------------ public
    def _retrieve(self, query: str) -> Tuple[Document, ...]:
        fused = self._rrf([self._dense(query), self._sparse(query)])
        if self.reranker is not None:
            fused = self._rerank(query, fused[: self.rerank_pool])
        return tuple(fused[: self.top_k])

    def retrieve(self, query: str) -> List[Document]:
        return list(self._cached(query.strip()))

    # LangChain-style alias so existing code like `retriever.invoke(q)` still works
    def invoke(self, query: str) -> List[Document]:
        return self.retrieve(query)

    def debug(self, query: str) -> Dict[str, List[str]]:
        """Show what each stage returned (great for demos and for your interview)."""
        short = lambda docs: [d.page_content[:80].replace("\n", " ") for d in docs]
        dense, sparse = self._dense(query), self._sparse(query)
        fused = self._rrf([dense, sparse])
        reranked = self._rerank(query, fused[: self.rerank_pool]) if self.reranker else fused
        return {
            "dense_top": short(dense[: self.top_k]),
            "bm25_top": short(sparse[: self.top_k]),
            "fused_top": short(fused[: self.top_k]),
            "final_top": short(reranked[: self.top_k]),
        }
